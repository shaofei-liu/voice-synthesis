import torch
from pathlib import Path
from config import TTS_MODEL_NAME, SAMPLE_RATE, LANGUAGE_OPTIONS, OUTPUT_DIR, MODELS_DIR
from audio_utils import preprocess_reference_audio, save_audio
import os
import sys
import io

# Set TTS model cache directory to local models folder
os.environ["TTS_HOME"] = str(MODELS_DIR)

# PyTorch 2.6+ compatibility fix - disable weights_only safety check for TTS model loading
torch_original_load = torch.load

def torch_load_permissive(*args, **kwargs):
    """Permissive torch.load that disables weights_only for TTS model loading"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return torch_original_load(*args, **kwargs)

torch.load = torch_load_permissive

# Auto-accept TTS license by providing stdin
original_stdin = sys.stdin
sys.stdin = io.StringIO("y\n")

try:
    # Import TTS with compatibility fixes
    try:
        from transformers import pytorch_utils
        if not hasattr(pytorch_utils, 'isin_mps_friendly'):
            pytorch_utils.isin_mps_friendly = lambda *args, **kwargs: False
    except:
        pass
    
    from TTS.api import TTS
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Error: Failed to import TTS: {e}")
    TTS_AVAILABLE = False
    TTS = None
finally:
    # Restore original stdin
    sys.stdin = original_stdin


class TTSEngine:
    """TTS Speech Synthesis Engine"""
    
    def __init__(self):
        """Initialize TTS Model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tts = None
        self.model_loaded = False
        
        if not TTS_AVAILABLE:
            raise RuntimeError("TTS library is not available! Please check installation")
        
        try:
            # Temporarily replace stdin to auto-accept license
            old_stdin = sys.stdin
            sys.stdin = io.StringIO("y\n")
            try:
                # Disable progress bar and enable CPU optimization for faster inference
                self.tts = TTS(
                    model_name=TTS_MODEL_NAME,
                    gpu=(self.device == "cuda"),
                    progress_bar=False  # Disable progress bar to reduce CPU overhead
                )
                self.model_loaded = True
                print(f"✅ TTS Model loaded successfully ({self.device})")
            finally:
                sys.stdin = old_stdin
        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model: {str(e)}")
    
    def is_ready(self):
        """Check if model is ready"""
        return self.model_loaded and self.tts is not None
    
    def synthesize(self, text, reference_audio_path, language_code="en", output_path=None):
        """
        Synthesize speech
        
        Args:
            text: Input text
            reference_audio_path: Reference audio path (for voice cloning)
            language_code: Language code
            output_path: Output path
            
        Returns:
            (audio_data, sample_rate, output_path)
        """
        if not self.is_ready():
            raise RuntimeError("TTS model is not ready")
        
        if not Path(reference_audio_path).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        
        print(f"Synthesizing: {text[:50]}...")
        
        if not output_path:
            output_path = OUTPUT_DIR / f"synthesis_{hash(text)}.wav"
        
        # Use XTTS v2 for voice cloning synthesis with continuous text processing
        # Optimizations: split_sentences=False for faster processing
        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio_path,
            language=language_code,
            file_path=str(output_path),
            split_sentences=False  # Faster: Process full text without sentence splitting
        )
        
        # Load generated audio
        import librosa
        audio, sample_rate = librosa.load(str(output_path), sr=SAMPLE_RATE)
        
        print(f"✅ Synthesis complete! Output: {output_path}")
        return audio, sample_rate, str(output_path)
    
    def batch_synthesize(self, texts, reference_audio_path, language_code="en"):
        """
        Batch synthesize speech
        
        Args:
            texts: List of texts
            reference_audio_path: Reference audio path
            language_code: Language code
            
        Returns:
            List of output paths
        """
        output_paths = []
        
        for i, text in enumerate(texts):
            output_file = f"batch_output_{i}.wav"
            output_path = OUTPUT_DIR / output_file
            try:
                _, _, output_path = self.synthesize(
                    text, 
                    reference_audio_path, 
                    language_code,
                    str(output_path)
                )
                output_paths.append(output_path)
            except Exception as e:
                print(f"❌ Failed to synthesize item {i+1}: {str(e)}")
        
        return output_paths
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return LANGUAGE_OPTIONS


# Global TTS Engine instance
_tts_engine = None


def get_tts_engine():
    """Get TTS Engine singleton"""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine
