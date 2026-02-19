# Voice Synthesis with XTTS v2 Voice Cloning

A FastAPI-based text-to-speech service featuring multilingual voice cloning using Coqui TTS's XTTS v2 model.

## Features

✨ **Voice Cloning**: Clone voices from custom audio files or use predefined sample voices  
🌍 **Multilingual**: Support for English and German (and other languages via XTTS v2)  
🎙️ **Sample Voices**: 11 pre-configured sample voices (5 English + 6 German)  
⚡ **Fast Inference**: CPU-optimized with model pre-loading on startup  
🐳 **Docker Ready**: Easy deployment with Docker containerization  
📦 **RESTful API**: Simple HTTP endpoints for integration  

## Sample Voices

### English Voices (5)
- Donald Trump
- Elon Musk
- Harry Kane
- Morgan Freeman
- Taylor Swift

### German Voices (6)
- Angela Merkel
- Anke Engelke
- Günther Jauch
- Heiner Lauterbach
- Herbert Grönemeyer
- Thomas Müller

## Installation

### Requirements
- Python 3.11+
- FFmpeg (for audio format conversion)
- 8GB+ RAM (model download and inference)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/shaofei-liu/voice-synthesis.git
cd voice-synthesis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

The API will be available at `http://localhost:7860`

## API Usage

### Get Languages
```bash
GET /languages
```

### Get Available Sample Voices
```bash
GET /samples
```

Response:
```json
{
  "en": {
    "language": "English",
    "voices": [
      {
        "filename": "donald_trump.wav",
        "name": "Donald Trump"
      },
      ...
    ]
  },
  "de": {
    "language": "German",
    "voices": [...]
  }
}
```

### Synthesize Speech (with Sample Voice)
```bash
POST /synthesize
```

Form Data:
- `text`: Text to synthesize (required)
- `language`: Language code - "en" or "de" (required)
- `sample_audio`: Sample voice filename (e.g., "donald_trump.wav") (optional)

Example:
```python
import requests

url = "http://localhost:7860/synthesize"
data = {
    "text": "Hello, this is a test",
    "language": "en",
    "sample_audio": "donald_trump.wav"
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
│   ├── donald_trump.wav
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
