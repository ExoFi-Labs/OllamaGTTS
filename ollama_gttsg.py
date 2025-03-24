import ollama
from gtts import gTTS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tempfile
import threading
import time
import queue
import re
import subprocess
import json
import numpy as np
import torch
import pyaudio
import wave
from faster_whisper import WhisperModel

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# Queues for handling audio playback and processing
audio_queue = queue.Queue()
cleanup_files = []  # Store files that couldn't be deleted immediately
is_speaking = False
playback_thread = None

# Add at the top with other global variables
tts_sequence = 0  # Global counter for ordering TTS chunks
tts_queue = queue.PriorityQueue()  # Queue for ordering TTS chunks

# Add these global variables after the other globals
should_stop_speaking = False
response_generator = None

# Initialize pygame mixer with settings that reduce file locking
pygame.mixer.quit()  # Reset mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)  # Smaller buffer for faster response

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# FFmpeg availability flag - set once at startup
has_ffmpeg = check_ffmpeg()

# Check for GPU availability
def check_gpu_availability():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# Device for torch/whisper operations
device = check_gpu_availability()
print(f"Using device: {device}")

class SpeechListener:
    def __init__(self, transcription_callback, vad_threshold=0.4, silence_duration=1.2, buffer_duration=0.65):
        # Initialize audio parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 512  # Small chunks for VAD
        
        # VAD parameters
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.buffer_duration = buffer_duration  # Buffer duration in seconds
        
        # Load the VAD model
        print("Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False
        )
        self.vad_model.to(device)
        print("VAD model loaded.")
        
        # Whisper model - use GPU if available
        print("Loading Whisper model...")
        # Determine appropriate compute type based on device capabilities
        compute_type = "int8"  # Always use int8 regardless of device
        self.whisper_model = WhisperModel(
            "base", 
            device=device, 
            compute_type=compute_type
        )
        print("Whisper model loaded.")
        
        # Callback function to handle transcription
        self.transcription_callback = transcription_callback
        
        # Threading and state management
        self.audio_queue = queue.Queue()
        self.recording_data = []
        self.is_recording = False
        self.running = False
        self.last_speech_time = 0
        self.audio_stream = None
        self.pyaudio_instance = None
        self.buffer_data = []  # Buffer to store initial audio data
        
    def start(self):
        self.running = True
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop(self):
        self.running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
        
    def _process_audio(self):
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.5)
                if data:
                    # Convert audio data for VAD processing
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    
                    # Run VAD on audio chunk
                    tensor = torch.from_numpy(audio_float).to(device)
                    speech_prob = self.vad_model(tensor, self.RATE).item()
                    current_time = time.time()
                    
                    if speech_prob > self.vad_threshold:
                        # Speech detected
                        self.last_speech_time = current_time
                        if not self.is_recording:
                            self.is_recording = True
                            print("\nSpeech detected, listening...")
                            # Add buffer data to recording data
                            self.recording_data.extend(self.buffer_data)
                            self.buffer_data = []
                        self.recording_data.append(data)
                    else:
                        # No speech detected
                        if self.is_recording:
                            # If silence has been detected for longer than silence_duration
                            if current_time - self.last_speech_time > self.silence_duration:
                                self.is_recording = False
                                print("Speech ended, transcribing...")
                                self._transcribe_recording()
                            else:
                                # Still within silence tolerance, keep recording
                                self.recording_data.append(data)
                        else:
                            # Store data in buffer if not recording
                            self.buffer_data.append(data)
                            # Limit buffer size to buffer_duration
                            max_buffer_size = int(self.buffer_duration * self.RATE / self.CHUNK)
                            if len(self.buffer_data) > max_buffer_size:
                                self.buffer_data.pop(0)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Audio processing error: {str(e)}")
                
    def _transcribe_recording(self):
        if not self.recording_data:
            return
            
        # Convert recorded audio data to a single numpy array
        audio_data = b''.join(self.recording_data)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Reset recording buffer
        self.recording_data = []
        
        # Run transcription in a separate thread to avoid blocking
        threading.Thread(
            target=self._run_transcription, 
            args=(audio_np,), 
            daemon=True
        ).start()
        
    def _run_transcription(self, audio_np):
        try:
            # Save audio to a temporary file for Whisper
            temp_file = os.path.join(tempfile.gettempdir(), f"temp_recording_{int(time.time())}.wav")
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.RATE)
                wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())
            
            # Run transcription
            segments, info = self.whisper_model.transcribe(temp_file, language="en", beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
            
            # Call the callback with the transcription
            if transcription.strip():
                self.transcription_callback(transcription)
                
            # Clean up
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                if temp_file not in cleanup_files:
                    cleanup_files.append(temp_file)
                
        except Exception as e:
            print(f"Transcription error: {str(e)}")

class ConversationManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.memory_file = "conversation_history.json"
        self.load_history()
    
    def add_exchange(self, user_input, assistant_response):
        """Add a new conversation exchange to history"""
        exchange = {
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': time.time()
        }
        self.history.append(exchange)
        
        # Keep history within max limit
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Save to file
        self.save_history()
    
    def get_context_string(self):
        """Convert history to a string format for the model"""
        context = []
        for exchange in self.history:
            context.append(f"User: {exchange['user']}")
            context.append(f"Assistant: {exchange['assistant']}")
        return "\n".join(context)
    
    def save_history(self):
        """Save conversation history to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def load_history(self):
        """Load conversation history from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Error loading conversation history: {e}")
            self.history = []

def cleanup_temp_files():
    """Attempt to clean up any leftover temporary files"""
    global cleanup_files
    for file in list(cleanup_files):
        try:
            if os.path.exists(file):
                os.remove(file)
                cleanup_files.remove(file)
                print(f"Cleaned up file: {file}")
        except Exception:
            pass  # We'll try again later

def interrupt_speech():
    """Interrupt current speech and response generation"""
    global should_stop_speaking, response_generator, is_speaking, tts_sequence
    should_stop_speaking = True
    
    # Force stop pygame audio
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    
    # Clear the TTS queue
    while not tts_queue.empty():
        try:
            _, temp_filename = tts_queue.get_nowait()
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                if temp_filename not in cleanup_files:
                    cleanup_files.append(temp_filename)
        except queue.Empty:
            break
    
    # Reset sequence counter to avoid orphaned audio files
    tts_sequence = 0
    is_speaking = False

def play_audio_from_queue():
    """Worker function to play audio files from the queue"""
    global is_speaking, cleanup_files, should_stop_speaking
    next_sequence = 0
    
    while True:
        if should_stop_speaking:
            # Complete interruption handling
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
            # Clear queue
            while not tts_queue.empty():
                try:
                    _, temp_filename = tts_queue.get_nowait()
                    try:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                    except:
                        if temp_filename not in cleanup_files:
                            cleanup_files.append(temp_filename)
                except queue.Empty:
                    break
            
            next_sequence = 0
            is_speaking = False
            should_stop_speaking = False
            time.sleep(0.1)
            continue

        try:
            if not tts_queue.empty():
                sequence, temp_filename = tts_queue.queue[0]  # Peek at next item
                
                if sequence == next_sequence:
                    sequence, temp_filename = tts_queue.get()
                    is_speaking = True
                    
                    try:
                        if len(cleanup_files) > 0 and not pygame.mixer.music.get_busy():
                            cleanup_temp_files()
                        
                        if should_stop_speaking:  # Check again before loading
                            continue
                            
                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()
                        
                        while pygame.mixer.music.get_busy() and not should_stop_speaking:
                            pygame.time.wait(50)  # Reduced wait time for faster response
                        
                        pygame.mixer.music.unload()
                        
                    except Exception as e:
                        print(f"Audio playback error: {str(e)}")
                    finally:
                        try:
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
                        except:
                            if temp_filename not in cleanup_files:
                                cleanup_files.append(temp_filename)
                        
                        if not should_stop_speaking:
                            next_sequence += 1
                        is_speaking = False
            
            time.sleep(0.05)  # Reduced sleep time for more responsive interruption
        except Exception:
            time.sleep(0.05)

def create_and_queue_audio(text, lang='en'):
    """Create audio file and add it to playback queue"""
    if not text.strip():
        return
    
    global tts_sequence, has_ffmpeg
    current_sequence = tts_sequence
    tts_sequence += 1
    
    def process_audio():
        try:
            # Create a temporary file with a simple name pattern
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time() * 1000)
            temp_filename = os.path.join(temp_dir, f"tts_{timestamp}.mp3")

            # Generate and save the audio (set to English voice)
            tts = gTTS(text=text, lang=lang, tld='co.uk', slow=False)
            tts.save(temp_filename)

            # Apply speed-up effect using FFmpeg if available
            if has_ffmpeg:
                try:
                    sped_up_filename = temp_filename.replace(".mp3", "_sped.mp3")
                    speed_factor = 1.15

                    subprocess.run([
                        "ffmpeg", "-y", "-i", temp_filename, "-filter:a", f"atempo={speed_factor}",
                        "-b:a", "192k", sped_up_filename
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    os.remove(temp_filename)  # Remove the original slower file
                    temp_filename = sped_up_filename
                except Exception as e:
                    print(f"FFmpeg processing error (using normal speed): {str(e)}")
                    # Continue with the original file if FFmpeg fails

            # Add to ordered queue with sequence number
            tts_queue.put((current_sequence, temp_filename))

        except Exception as e:
            print(f"TTS error: {str(e)}")
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except:
                pass

    # Start audio processing in a separate thread
    audio_thread = threading.Thread(target=process_audio, daemon=True)
    audio_thread.start()

def start_playback_thread():
    """Initialize and start the audio playback thread"""
    global playback_thread
    if (playback_thread is None or not playback_thread.is_alive()):
        playback_thread = threading.Thread(target=play_audio_from_queue, daemon=True)
        playback_thread.start()

def get_available_models():
    """Get list of models available locally through Ollama using CLI command"""
    try:
        # Use the command-line tool instead of the Python API
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running ollama list: {result.stderr}")
            return []
            
        # Parse the output
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Just the header or empty
            return []
            
        # Skip the header line
        model_lines = lines[1:]
        models = []
        
        for line in model_lines:
            # Split by whitespace, but handle multiple spaces
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 3:
                models.append({
                    'name': parts[0].strip(),
                    'id': parts[1].strip() if len(parts) > 1 else "",
                    'size': parts[2].strip() if len(parts) > 2 else "",
                    'modified': ' '.join(parts[3:]) if len(parts) > 3 else ""
                })
                
        return models
        
    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return []

def select_model():
    """Display available models and let user select one"""
    models = get_available_models()
    
    if not models:
        print("No models found. Make sure Ollama is running.")
        return "gemma:2b"  # Default fallback
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        name = model.get('name', 'Unknown')
        size = model.get('size', 'Unknown size')
        modified = model.get('modified', 'Unknown date')
        print(f"{i}. {name} ({size}) - Last modified: {modified}")
    
    while True:
        try:
            choice = input("\nSelect a model number (or press Enter for default): ").strip()
            if not choice:
                return models[0]['name']  # First model as default
            
            choice = int(choice)
            if 1 <= choice <= len(models):
                selected_model = models[choice-1]['name']
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except Exception as e:
            print(f"Error selecting model: {str(e)}")
            if models:
                return models[0]['name']  # First model as fallback
            return "gemma:2b"  # Default fallback

def process_text_for_tts(text):
    """Process text to ensure better TTS quality and remove special characters"""
    # Remove unwanted special characters
    text = re.sub(r'[*<>{}()\[\]&%#@^_=+~]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    # Handle common abbreviations (e.g., "U.S." -> "U S")
    text = re.sub(r'(\w)\.(\w)\.', r'\1 \2 ', text)

    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)

    return text

def smart_chunk_text(text, min_length=30, max_length=150):
    """Split text into natural chunks for TTS processing using sentence boundaries"""
    # First try to split by sentences
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        sentence_length = len(sentence)
        
        # If a single sentence is longer than max_length, we'll need to split it further
        if sentence_length > max_length:
            # Try to split on commas or other natural pauses
            clause_pattern = re.compile(r'(?<=[,;:])\s+')
            clauses = re.split(clause_pattern, sentence)
            
            for clause in clauses:
                if not clause.strip():
                    continue
                    
                clause_length = len(clause)
                
                # If the clause fits or the current chunk is empty, add it
                if current_length + clause_length <= max_length or not current_chunk:
                    current_chunk.append(clause)
                    current_length += clause_length
                else:
                    # Process the current chunk if it's long enough
                    if current_length >= min_length:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [clause]
                        current_length = clause_length
                    else:
                        # Otherwise, keep adding despite going over max_length
                        current_chunk.append(clause)
                        current_length += clause_length
        else:
            # If adding this sentence doesn't exceed max_length or the chunk is empty, add it
            if current_length + sentence_length <= max_length or not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Process the current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_response_chunk(text_chunk):
    """Process a single chunk of text for TTS"""
    if not text_chunk.strip():
        return
    processed_text = process_text_for_tts(text_chunk)
    create_and_queue_audio(processed_text)
    # Remove the audio_queue.join() to allow for parallel processing

def get_model_default_system_prompt(model_name):
    """Get the default system prompt for a model if it exists"""
    try:
        # Use ollama show command to get model info
        result = subprocess.run(['ollama', 'show', 'system', model_name], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except Exception as e:
        print(f"Error getting default system prompt: {str(e)}")
        return None

def handle_transcription(text):
    global should_stop_speaking, response_generator
    
    # Interrupt current speech if any
    interrupt_speech()
    
    if text.lower() in ('exit', 'quit', 'goodbye'):
        print("\nDetected exit command.")
        return False
    
    print(f"\nYou: {text}")
    
    # Create context-aware prompt
    context = conversation.get_context_string()
    context_prompt = f"{system_message}\n{context}\n\nUser: {text}\nAssistant:"
    
    # Print assistant response header
    print("\nAssistant: ", end="", flush=True)
    
    # Generate streaming response
    full_response = ""
    current_text = ""
    buffer = []
    
    # Stream the response
    response_generator = ollama.generate(
        model=selected_model,
        prompt=text,
        system=context_prompt,
        stream=True
    )
    
    try:
        for chunk in response_generator:
            if should_stop_speaking:
                break
                
            # Get the new token
            new_text = chunk['response']
            print(new_text, end="", flush=True)
            
            current_text += new_text
            full_response += new_text
            
            # Check for natural breakpoints and process TTS
            if re.search(r'[.!?]\s*$', current_text) and len(current_text) >= 30:
                buffer.append(current_text)
                if len(buffer) >= 1:
                    combined_text = " ".join(buffer)
                    process_response_chunk(combined_text)
                    buffer = []
                current_text = ""
                
        # Process any remaining text
        if not should_stop_speaking and (buffer or current_text.strip()):
            remaining_text = " ".join(buffer + [current_text.strip()])
            if remaining_text.strip():
                process_response_chunk(remaining_text)
    
    except Exception as e:
        print(f"\nError during response generation: {e}")
    finally:
        response_generator = None
        should_stop_speaking = False
    
    # Add the exchange to conversation history only if we have a complete response
    if full_response.strip():
        conversation.add_exchange(text, full_response)
    
    print("")  # New line after response
    return True

def main():
    # Start the audio playback thread
    start_playback_thread()
    
    # Initial setup
    print("Welcome to Ollama Voice Assistant!")
    
    # Check if FFmpeg is available and inform user
    global has_ffmpeg
    if has_ffmpeg:
        print("FFmpeg found - Audio speed adjustment enabled")
    else:
        print("FFmpeg not found - Audio will play at normal speed")
    
    create_and_queue_audio("Welcome to Ollama Voice Assistant!")
    
    # Initialize conversation manager
    conversation = ConversationManager()
    
    # Let user select a model
    selected_model = select_model()
    print(f"\nUsing model: {selected_model}")
    create_and_queue_audio(f"Using model {selected_model}")
    
    # Try to get model's default system prompt
    default_system_message = get_model_default_system_prompt(selected_model)
    if default_system_message:
        print(f"\nModel's default system prompt available.")
    
    # Get user input for system message
    system_message = input("Enter system message (or press Enter to use model's default): ").strip()
    # Use default if user skips and default exists, otherwise use generic default
    system_message = system_message or default_system_message or "You are a helpful assistant."
    
    # Add memory context to system message
    system_message += "\nHere's our conversation history for context:\n"
    
    # Initialize the speech listener with transcription handler
    def handle_transcription(text):
        interrupt_speech()  # Interrupt ongoing response if any
        if text.lower() in ('exit', 'quit', 'goodbye'):
            print("\nDetected exit command.")
            return False
        
        print(f"\nYou: {text}")
        
        # Create context-aware prompt
        context = conversation.get_context_string()
        context_prompt = f"{system_message}\n{context}\n\nUser: {text}\nAssistant:"
        
        # Print assistant response header
        print("\nAssistant: ", end="", flush=True)
        
        # Generate streaming response
        full_response = ""
        current_text = ""
        buffer = []
        
        # Stream the response
        for chunk in ollama.generate(
            model=selected_model,
            prompt=text,
            system=context_prompt,
            stream=True
        ):
            # Get the new token
            new_text = chunk['response']
            print(new_text, end="", flush=True)
            
            current_text += new_text
            full_response += new_text
            
            # Check for natural breakpoints and process TTS
            if re.search(r'[.!?]\s*$', current_text) and len(current_text) >= 30:
                buffer.append(current_text)
                if len(buffer) >= 1:
                    combined_text = " ".join(buffer)
                    process_response_chunk(combined_text)
                    buffer = []
                current_text = ""
        
        # Process any remaining text
        if buffer or current_text.strip():
            remaining_text = " ".join(buffer + [current_text.strip()])
            if remaining_text.strip():
                process_response_chunk(remaining_text)
        
        # Add the exchange to conversation history
        conversation.add_exchange(text, full_response)
        
        print("")  # New line after response
        return True
    
    # Initialize and start speech listener
    speech_listener = SpeechListener(
        transcription_callback=handle_transcription, 
        vad_threshold=0.4,  # Adjust sensitivity (0.3-0.7 range works well)
        silence_duration=1.2,  # Wait 1 second of silence before processing
        buffer_duration=0.65  # Buffer duration in seconds
    )
    speech_listener.start()
    
    print("\nVoice detection active. Speak, or type a message and press Enter. Type 'quit' to exit.")
    
    try:
        # Allow for typed input as well as voice
        while True:
            typed_input = input("You: ")  # Modified input prompt
            if typed_input.lower() in ('exit', 'quit', 'goodbye'):
                print("\nGoodbye!")
                create_and_queue_audio("Goodbye!")
                break
            elif typed_input:
                if not handle_transcription(typed_input):
                    break
    
    except KeyboardInterrupt:
        interrupt_speech()
        print("\nGoodbye!")
        create_and_queue_audio("Goodbye!")
    finally:
        # Wait for final audio to finish
        time.sleep(2)
        speech_listener.stop()
        cleanup_temp_files()

if __name__ == "__main__":
    # Register an exit handler to cleanup temp files
    import atexit
    atexit.register(cleanup_temp_files)
    main()