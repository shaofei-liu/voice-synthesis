import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Directory paths
UPLOADED_FILES_DIR = PROJECT_ROOT / "uploaded_files"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Create necessary directories
for dir_path in [UPLOADED_FILES_DIR, OUTPUT_DIR, MODELS_DIR, SAMPLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Audio configuration
SAMPLE_RATE = 22050
AUDIO_DURATION_SECONDS = 30
AUDIO_FORMAT = "wav"

# TTS model configuration
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE_OPTIONS = {
    "en": "English",
    "de": "German",
}

# XTTS v2 synthesis parameters - per language optimization
# These control the balance between voice consistency and naturalness
TTS_SYNTHESIS_PARAMS = {
    # English parameters - slightly less conservative for more natural flow
    "en": {
        "temperature": 0.52,    # Conservative but slightly less strict than German
                                # English is more forgiving with slight variation
        "top_p": 0.68,          # Slightly more diverse than German
        "top_k": 35,            # Slightly higher than German for better naturalness
        "split_sentences": False,
        "speech_rate": 0.85,
    },
    # German parameters - very conservative for clean end-of-utterance
    "de": {
        "temperature": 0.50,    # Most conservative - German is strict about precision
        "top_p": 0.65,          # Very conservative nucleus sampling
        "top_k": 30,            # Conservative to avoid high-freq artifacts
        "split_sentences": False,
        "speech_rate": 0.85,
    },
    # Lowpass filter frequencies per language
    "lowpass_freq": {
        "en": 8500,             # English can handle slightly more high-freq
        "de": 8000,             # German prefers lower cutoff for clarity
    }
}

# Sample audio files organized by language
SAMPLE_AUDIOS = {
    "en": [
        "donald_trump.wav",
        "elon_musk.wav",
        "harry_kane.wav",
        "morgan_freeman.wav",
        "taylor_swift.wav",
    ],
    "de": [
        "anke_engelke.wav",
        "günther_jauch.wav",
        "heiner_lauterbach.wav",
        "herbert_grönemeyer.wav",
        "thomas_müller.wav",
    ]
}

# Display names for sample voices
SAMPLE_VOICE_NAMES = {
    "donald_trump.wav": "Donald Trump",
    "elon_musk.wav": "Elon Musk",
    "harry_kane.wav": "Harry Kane",
    "morgan_freeman.wav": "Morgan Freeman",
    "taylor_swift.wav": "Taylor Swift",
    "anke_engelke.wav": "Anke Engelke",
    "günther_jauch.wav": "Günther Jauch",
    "heiner_lauterbach.wav": "Heiner Lauterbach",
    "herbert_grönemeyer.wav": "Herbert Grönemeyer",
    "thomas_müller.wav": "Thomas Müller",
}

# FastAPI configuration
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
REQUEST_TIMEOUT = 300  # 5 minutes timeout
