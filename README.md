# OllamaGTTS - Voice Assistant for Ollama

A lightweight voice assistant that uses Ollama for generating responses and Google Text-to-Speech (gTTS) for speaking those responses aloud. This application allows for interactive voice conversations with any Ollama model installed on your system.

## Features

- Text-to-speech output for Ollama responses using Google's TTS service
- Conversation memory that persists between sessions
- Smart text chunking for natural speech pauses
- Optional audio speed adjustment using FFmpeg (if installed)
- Works with all Ollama models
- Model selection interface
- Custom system prompts support
- Response Interruption Support

## Requirements

- Python 3.7+
- Ollama installed and running locally
- Internet connection (for Google TTS service)
- FFmpeg (optional - for audio speed adjustment)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ExoFi-Labs/OllamaGTTS.git
cd OllamaGTTS
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install required Python packages

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

If you haven't already installed Ollama, follow the instructions at [Ollama's official website](https://ollama.ai/download).

Make sure you have at least one model downloaded using:

```bash
ollama pull llama3.3
```

or any other model of your choice.

### 5. (Optional) Install FFmpeg for audio speed adjustment

FFmpeg is used to adjust the speed of audio playback. The application works without FFmpeg, but audio will play at normal speed.

- **Windows**: Download from [FFmpeg's official website](https://ffmpeg.org/download.html) and add to your PATH
- **macOS**: Install using Homebrew: `brew install ffmpeg`
- **Ubuntu/Debian**: Install using apt: `sudo apt install ffmpeg`
- **Other Linux**: Use your distribution's package manager
- **Untested on Mac/Ubuntu**

## Usage

1. Run the application:

```bash
python ollama_gttsg.py
```

3. Select a model from the list of available models
4. Enter a system message or press Enter to use the model's default
5. Start your conversation with the assistant

### Commands

- Type your message and press Enter to send
- Type `exit` or `quit` to end the conversation

## How It Works

1. The application connects to your local Ollama instance and lists available models
2. When you send a message, it gets streamed to the selected Ollama model
3. As responses are received, the text is chunked at natural pause points
4. Each chunk is converted to speech using Google's TTS service
5. Audio is played back in the correct order with speed adjustment (if FFmpeg is available)
6. Conversation history is stored for future context

## Customization

### Changing the Voice

To change the TTS voice language, modify the `lang` parameter in the `create_and_queue_audio` function. Default is English ('en').

### Adjusting Speech Speed

If you have FFmpeg installed, you can change the speech speed by modifying the `speed_factor` value in the `create_and_queue_audio` function. The default is 1.15 (15% faster than normal).

## Troubleshooting

### No audio output

- Make sure your system's audio is not muted
- Check that pygame is properly installed
- Try restarting the application

### Models not showing up

- Verify that Ollama is running in the background
- Make sure you've downloaded at least one model using `ollama pull`

### FFmpeg not found

- If you want audio speed adjustment, make sure FFmpeg is installed and available in your system PATH
- Without FFmpeg, the application will still work but will use normal speed audio

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
