import ollama
from gtts import gTTS
import os
import tempfile
import threading
import time
import queue
import re
import subprocess
import sys
import json

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# Queue for handling audio playback
audio_queue = queue.Queue()
cleanup_files = []  # Store files that couldn't be deleted immediately
is_speaking = False
playback_thread = None

# Add at the top with other global variables
tts_sequence = 0  # Global counter for ordering TTS chunks
tts_queue = queue.PriorityQueue()  # Queue for ordering TTS chunks

# Initialize pygame mixer with settings that reduce file locking
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# FFmpeg availability flag - set once at startup
has_ffmpeg = check_ffmpeg()

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

def play_audio_from_queue():
    """Worker function to play audio files from the queue"""
    global is_speaking, cleanup_files
    next_sequence = 0
    
    while True:
        try:
            if not tts_queue.empty():
                # Peek at the next item's sequence number
                sequence, _ = tts_queue.queue[0]
                
                # Only process if it's the next in sequence
                if sequence == next_sequence:
                    sequence, temp_filename = tts_queue.get()
                    is_speaking = True
                    
                    try:
                        # Try to delete any previously failed files
                        if len(cleanup_files) > 0 and not pygame.mixer.music.get_busy():
                            cleanup_temp_files()
                            
                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()
                        
                        while pygame.mixer.music.get_busy():
                            pygame.time.wait(100)
                        
                        pygame.time.wait(200)
                        pygame.mixer.music.unload()
                        
                    except Exception as e:
                        print(f"Audio playback error: {str(e)}")
                    
                    try:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                    except Exception:
                        if temp_filename not in cleanup_files:
                            cleanup_files.append(temp_filename)
                    
                    next_sequence += 1
                    is_speaking = False
                    
            time.sleep(0.1)
        except Exception:
            time.sleep(0.1)

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
    if playback_thread is None or not playback_thread.is_alive():
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
    
    # Continuous conversation loop
    while True:
        try:
            prompt = input("\nYou: ").strip()

            if not prompt:
                continue
            if prompt.lower() in ('exit', 'quit'):
                print("Goodbye!")
                create_and_queue_audio("Goodbye!")
                # Wait for final audio to finish
                time.sleep(2)  # Give some time for the last audio to play
                # Final cleanup attempt
                cleanup_temp_files()
                break
            
            # Create context-aware prompt
            context = conversation.get_context_string()
            context_prompt = f"{system_message}\n{context}\n\nUser: {prompt}\nAssistant:"
            
            # Print assistant response header
            print("\nAssistant: ", end="", flush=True)
            
            # Generate streaming response
            full_response = ""
            current_text = ""
            buffer = []
            
            # Stream the response
            for chunk in ollama.generate(
                model=selected_model,
                prompt=prompt,
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
            conversation.add_exchange(prompt, full_response)
            
            # Give some time for audio to process before next prompt
            time.sleep(1)
            
            print("")  # New line after response
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            cleanup_temp_files()
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            create_and_queue_audio("Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    # Register an exit handler to cleanup temp files
    import atexit
    atexit.register(cleanup_temp_files)
    main()