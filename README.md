# Voice Synthesis with Voice Cloning

AI-powered Text-to-Speech application with multilingual voice cloning using XTTS v2.

## Features

- 🎤 **Voice Cloning**: Upload any audio file or use sample voices (English & German)
- 🌐 **Multilingual**: Supports English and German
- ⚡ **Real-time Synthesis**: Fast speech generation
- 🎨 **Modern Web Interface**: Responsive HTML/JS frontend
- 📁 **Multiple Audio Formats**: WAV, MP3, M4A, FLAC, OGG support

## Technical Details

- **TTS Model**: XTTS v2 (Coqui AI) - Multilingual zero-shot voice cloning
- **Backend**: FastAPI 0.104.1 + Uvicorn
- **Audio Processing**: librosa + FFmpeg + soundfile
- **Deep Learning**: PyTorch 2.10+ with CUDA support

## API Endpoints

### GET `/health`
Check if TTS model is ready

### GET `/languages`
Get supported languages (returns `{en, de}`)

### GET `/samples`
Get available sample audio files

### POST `/synthesize`
Synthesize speech with voice cloning

**Parameters:**
- `text` (string, required): Text to synthesize
- `language` (string): Language code (`en` or `de`, default: `en`)
- `reference_audio` (file, optional): Upload custom audio for voice cloning
- `sample_audio` (string, optional): Use sample audio (`en` or `de`)

**Response:** WAV audio file

### POST `/synthesize-batch`
Batch synthesize multiple texts (separated by newlines)

## Usage

1. **Using Sample Voices**:
   - Select a sample voice from the available options
   - Enter text in your chosen language
   - Click "Synthesize"

2. **Using Custom Voice**:
   - Upload your audio file (min 2 seconds, any format)
   - Enter text
   - Click "Synthesize"

3. **API Usage**:
   ```bash
   curl -X POST "http://localhost:7860/synthesize" \
     -F "text=Hello world" \
     -F "language=en" \
     -F "sample_audio=en"
   ```

## Model Information

- **XTTS v2**: Multilingual TTS model supporting voice cloning
- **Automatic Sentence Splitting**: Disabled for natural long-text synthesis
- **Audio Preprocessing**: Gentle silence trimming (top_db=50) to preserve voice characteristics
- **Device**: Automatically detects and uses CUDA GPU if available, falls back to CPU

## Installation & Setup

### Local Development

**Prerequisites:**
- Python 3.11+
- FFmpeg (for audio format conversion)
- 16GB+ RAM (model download ~1.87GB + inference)
- GPU recommended for faster performance

**Installation Steps:**

```bash
# Clone the repository
git clone https://github.com/shaofei-liu/voice-synthesis.git
cd voice-synthesis

# Create conda environment
conda create -n voice python=3.11
conda activate voice

# Install FFmpeg
conda install ffmpeg -y

# Install Python dependencies
pip install -r requirements.txt

# Run the development server
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`

### Docker Deployment

**Build the Docker image:**
```bash
docker build -t voice-synthesis .
```

**Run the container:**
```bash
docker run -p 7860:7860 --gpus all voice-synthesis
# Or without GPU:
docker run -p 7860:7860 voice-synthesis
```

The API will be available at `http://localhost:7860`

## Environment Variables

- `PORT`: Server port (default: 8000 for local development, 7860 for Docker/production)

## Performance Notes

- **First synthesis**: ~1-2 minutes (model initialization and download on first run)
- **Subsequent syntheses**: ~1-2 minutes depending on text length and system resources
- **Model size**: XTTS v2 is ~1.87GB, requires download on first initialization
- **Recommended**: GPU acceleration significantly improves performance
- **Note**: Performance depends heavily on system resources and text complexity

## License

This project uses:
- XTTS v2 by Coqui AI (Apache 2.0)
- FastAPI (MIT)
- PyTorch (BSD)
- librosa, SoundFile, and other open-source libraries

See LICENSE file for details.

## Links

- **Personal Website**: [https://www.shaofeiliu.com](https://www.shaofeiliu.com/)
- **Coqui TTS**: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- **XTTS v2 Model**: [https://huggingface.co/coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For questions and inquiries:

1. Visit my personal website: [https://www.shaofeiliu.com](https://www.shaofeiliu.com/)
2. Open an issue on GitHub
3. Check the TTS documentation: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)

---

**Last Updated**: February 2026 | **Version**: 1.0.0

Form Data:
- `text`: Text to synthesize (required)
- `language`: Language code - "en" or "de" (required)
- `sample_audio`: Sample voice filename (e.g., "arnold_schwarzenegger.wav") (optional)

Example:
```python
import requests

url = "http://localhost:7860/synthesize"
data = {
    "text": "Hello, this is a test",
    "language": "en",
    "sample_audio": "arnold_schwarzenegger.wav"
}

response = requests.post(url, data=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Synthesize Speech (with Custom Voice)
```bash
POST /synthesize
```

Form Data:
- `text`: Text to synthesize (required)
- `language`: Language code - "en" or "de" (required)
- `reference_audio`: Custom audio file (.wav, .mp3, .flac, etc.) (optional)

Example:
```python
import requests

url = "http://localhost:7860/synthesize"
files = {"reference_audio": open("my_voice.wav", "rb")}
data = {
    "text": "Clone my voice and say this",
    "language": "en"
}

response = requests.post(url, data=data, files=files)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Configuration

Edit `config.py` to customize:

```python
# Sample rate for audio processing
SAMPLE_RATE = 22050

# TTS model to use
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Maximum file upload size (50MB default)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# Request timeout (5 minutes default)
REQUEST_TIMEOUT = 300
```

## Docker Deployment

### Build Image
```bash
docker build -t voice-synthesis .
```

### Run Container
```bash
docker run -p 7860:7860 voice-synthesis
```

The API will be available at `http://localhost:7860`

## Performance Optimization

The application includes several optimizations:

1. **Model Pre-loading**: TTS model is loaded on startup for faster first request
2. **Progress Bar Disabled**: Reduces logging overhead during synthesis
3. **Continuous Text Processing**: Processes full text without splitting for speed
4. **CPU Memory Efficient**: Optimized for CPU inference with fp32 precision

## Architecture

### Components

- **tts_engine.py**: Core TTS engine with model management and voice synthesis
- **app.py**: FastAPI application with REST endpoints
- **audio_utils.py**: Audio processing utilities (loading, conversion, normalization)
- **config.py**: Configuration and constants

### Workflow

1. Request → FastAPI endpoint validation
2. Audio reference loading (sample or uploaded file)
3. Audio preprocessing (normalization, silence trimming)
4. TTS inference using XTTS v2 model
5. Audio file saving and response

## File Structure

```
voice-synthesis/
├── app.py                 # FastAPI application
├── tts_engine.py         # TTS model and synthesis logic
├── audio_utils.py        # Audio processing utilities
├── config.py             # Configuration
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
├── samples/              # Sample voice audio files (11 pre-configured)
│   ├── arnold_schwarzenegger.wav
│   ├── elon_musk.wav
│   └── ...
└── README.md             # This file
```

## Dependencies

- **FastAPI** (0.104.1): Web framework
- **Uvicorn** (0.24.0): ASGI server
- **Coqui TTS** (0.22.0): Text-to-speech engine with XTTS v2
- **PyTorch** (2.10+): Deep learning framework
- **Transformers** (4.36.0): NLP models
- **Librosa** (0.10.0): Audio processing
- **SoundFile** (0.13.1): Audio file I/O
- **torchcodec** (>=0.1.0): Audio codec support

## License

This project uses the Coqui TTS model which is available under the CPML (Non-Commercial Public Model License). Please review the license terms at https://coqui.ai/cpml before commercial use.

For commercial usage, you must obtain a commercial license from Coqui: licensing@coqui.ai

## Links

- **Live Demo**: https://www.shaofeiliu.com/#voice-synthesis
- **Personal Website**: https://www.shaofeiliu.com
- **Coqui TTS**: https://github.com/coqui-ai/TTS
- **XTTS v2 Model**: https://huggingface.co/coqui/XTTS-v2

## Contributing

Feel free to submit issues and enhancement requests!

## Support

For issues and questions:
1. Visit my personal website: https://www.shaofeiliu.com
2. Open an issue on GitHub
3. Check TTS documentation: https://github.com/coqui-ai/TTS

---

**Last Updated**: February 2026  
**Version**: 1.0.0
