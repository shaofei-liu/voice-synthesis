#!/usr/bin/env python
"""Quick MP3 to WAV converter using pydub fallback"""

from pathlib import Path
import subprocess
import sys

SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLE_RATE = 22050

mp3_file = SAMPLES_DIR / "arnold_schwarzenegger.mp3"
wav_file = SAMPLES_DIR / "arnold_schwarzenegger.wav"

if not mp3_file.exists():
    print(f"❌ File not found: {mp3_file}")
    sys.exit(1)

print(f"Converting {mp3_file.name} to WAV...")

# Try using Windows built-in tools or libraries
try:
    # Try pydub if available
    from pydub import AudioSegment
    
    print("  Loading MP3...")
    audio = AudioSegment.from_mp3(str(mp3_file))
    
    print("  Converting to WAV...")
    audio.export(str(wav_file), format="wav", codec="pcm_s16le", parameters=["-ar", str(SAMPLE_RATE)])
    
    size_kb = wav_file.stat().st_size / 1024
    print(f"✅ {wav_file.name} created ({size_kb:.1f} KB)")

except ImportError:
    print("  pydub not available, trying alternative method...")
    
    # Fallback: Use online conversion or manual instruction
    print("""
❌ Could not auto-convert. Please do one of the following:

Option 1: Online converter
  - Go to https://convertio.co/mp3-wav/
  - Upload: arnold_schwarzenegger.mp3
  - Set output to WAV, target sample rate 22050 Hz
  - Download arnold_schwarzenegger.wav
  - Save to: {SAMPLES_DIR}

Option 2: Use Audacity (free)
  - Open Audacity
  - File → Open → arnold_schwarzenegger.mp3
  - Select all (Ctrl+A)
  - Tracks → Resample → Set to 22050 Hz
  - File → Export → Export Audio as WAV
  - Save to: {SAMPLES_DIR}/arnold_schwarzenegger.wav

Option 3: Install FFmpeg and librosa
  - conda install ffmpeg librosa soundfile
  - python convert_samples.py
    """)
    sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
