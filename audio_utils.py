import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from config import SAMPLE_RATE, AUDIO_FORMAT
import io
import subprocess
import tempfile
import os


def load_audio(file_path):
    """
    Load audio file
    
    Args:
        file_path: File path or BytesIO object
        
    Returns:
        (audio_data, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return audio, sr
    except Exception as e:
        raise Exception(f"Failed to load audio: {str(e)}")


def normalize_audio(audio):
    """
    Normalize audio (volume normalization)
    
    Args:
        audio: Audio data
        
    Returns:
        Normalized audio data
    """
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    return audio


def trim_silence(audio, sr, top_db=40):
    """
    Remove silence from the beginning and end of audio
    
    Args:
        audio: Audio data
        sr: Sample rate
        top_db: Decibel threshold
        
    Returns:
        Trimmed audio
    """
    try:
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio
    except Exception as e:
        print(f"Failed to trim silence: {str(e)}")
        return audio


def save_audio(audio, file_path, sr=SAMPLE_RATE):
    """
    Save audio file
    
    Args:
        audio: Audio data
        file_path: Save path
        sr: Sample rate
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(file_path, audio, sr, subtype='PCM_16')
        return True
    except Exception as e:
        raise Exception(f"Failed to save audio: {str(e)}")


def preprocess_reference_audio(audio, sr):
    """
    Preprocess reference audio for voice cloning
    
    Args:
        audio: Audio data
        sr: Sample rate
        
    Returns:
        Preprocessed audio
    """
    # 1. Gentle silence trimming (less aggressive - top_db=50 instead of 30)
    audio = trim_silence(audio, sr, top_db=50)
    
    # 2. Normalize volume
    audio = normalize_audio(audio)
    
    # 3. Limit length (5-30 seconds max)
    max_samples = sr * 30
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    
    # 4. Minimum length check (at least 2 seconds)
    min_samples = sr * 2
    if len(audio) < min_samples:
        raise ValueError("Reference audio too short - minimum 2 seconds required")
    
    return audio, sr


def get_audio_duration(audio, sr):
    """
    Get audio duration in seconds
    
    Args:
        audio: Audio data
        sr: Sample rate
        
    Returns:
        Duration in seconds
    """
    return len(audio) / sr


def bytesio_to_wav(uploaded_file):
    """
    Convert uploaded file to WAV format
    
    Args:
        uploaded_file: Uploaded file object
        
    Returns:
        (audio_data, sample_rate)
    """
    try:
        # Read file content
        audio_bytes = uploaded_file.read()
        
        # Use librosa to load
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
        
        return audio, sr
    except Exception as e:
        raise Exception(f"File conversion failed: {str(e)}")


def convert_to_wav_ffmpeg(input_path, output_path=None):
    """
    Convert audio of any format to WAV using FFmpeg
    
    Args:
        input_path: Input audio file path
        output_path: Output WAV file path (optional)
        
    Returns:
        Output WAV file path
    """
    if output_path is None:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = temp_wav.name
        temp_wav.close()
    
    try:
        # Call ffmpeg directly (assuming installed via conda install ffmpeg)
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr[:200]}")
        
        return output_path
        
    except FileNotFoundError:
        raise Exception("FFmpeg not installed! Please run: conda install ffmpeg -y")
    except Exception as e:
        raise Exception(f"FFmpeg conversion failed: {str(e)}")
