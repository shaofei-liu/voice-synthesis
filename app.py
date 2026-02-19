from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import traceback
import os

from config import OUTPUT_DIR, SAMPLE_RATE, LANGUAGE_OPTIONS, REQUEST_TIMEOUT, SAMPLES_DIR, SAMPLE_AUDIOS, SAMPLE_VOICE_NAMES
from audio_utils import bytesio_to_wav, preprocess_reference_audio, save_audio, get_audio_duration, convert_to_wav_ffmpeg
from tts_engine import get_tts_engine

app = FastAPI(title="Voice Synthesis API")

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load TTS model on startup for faster inference
@app.on_event("startup")
async def startup_event():
    """Initialize and pre-load TTS model on application startup"""
    print("🚀 Initializing TTS model on startup...")
    try:
        engine = get_tts_engine()
        if engine.is_ready():
            print("✅ TTS model pre-loaded successfully")
        else:
            print("⚠️ TTS model initialization incomplete")
    except Exception as e:
        print(f"⚠️ Warning: TTS model pre-loading encountered an issue: {e}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Voice Synthesis API"}


@app.get("/health")
async def health_check():
    """Check if TTS model is ready"""
    engine = get_tts_engine()
    return {
        "status": "ready" if engine.is_ready() else "loading",
        "model_loaded": engine.is_ready()
    }


@app.get("/languages")
async def get_languages():
    """Get supported languages"""
    return {
        "languages": LANGUAGE_OPTIONS,
        "total": len(LANGUAGE_OPTIONS)
    }


@app.get("/samples")
async def get_samples():
    """Get available sample audio files organized by language"""
    samples = {}
    for lang, filenames in SAMPLE_AUDIOS.items():
        samples[lang] = {
            "language": LANGUAGE_OPTIONS.get(lang),
            "voices": []
        }
        for filename in filenames:
            filepath = SAMPLES_DIR / filename
            # Only include samples that actually exist
            if filepath.exists():
                samples[lang]["voices"].append({
                    "filename": filename,
                    "name": SAMPLE_VOICE_NAMES.get(filename, filename)
                })
    return samples


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form(default="en"),
    reference_audio: UploadFile = None,
    sample_audio: str = Form(None)
):
    """
    Synthesize speech with voice cloning
    
    Parameters:
    - text: Text to synthesize
    - language: Language code (en=English, de=German)
    - reference_audio: Reference audio file for voice cloning
    - sample_audio: Use a sample audio file (optional)
    
    Returns:
    - WAV audio file with synthesized speech
    """
    
    # Validate text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text length cannot exceed 5000 characters")
    
    # Validate language
    if language not in LANGUAGE_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported languages: {list(LANGUAGE_OPTIONS.keys())}"
        )
    
    # Check if using sample audio or uploaded audio
    audio_source = None
    if sample_audio:
        # Check if sample_audio is in any language's sample list
        sample_found = False
        for lang_samples in SAMPLE_AUDIOS.values():
            if sample_audio in lang_samples:
                sample_found = True
                break
        if sample_found:
            audio_source = "sample"
    
    if audio_source is None and reference_audio:
        audio_source = "upload"
    elif audio_source is None:
        raise HTTPException(status_code=400, detail="Please provide either a reference audio file or select a sample audio")
    
    # If using uploaded file, validate it
    if audio_source == "upload":
        allowed_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        file_ext = Path(reference_audio.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Supported formats: {', '.join(allowed_extensions)}"
            )
    
    try:
        # Get TTS engine
        engine = get_tts_engine()
        
        # Variables for cleanup
        temp_audio_files = []
        ref_audio_path = None
        
        # Handle reference audio
        try:
            # Determine audio source
            if audio_source == "sample":
                ref_audio_path = SAMPLES_DIR / sample_audio
                if not ref_audio_path.exists():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Sample audio file not found: {sample_audio}"
                    )
                import librosa
                ref_audio, ref_sr = librosa.load(str(ref_audio_path), sr=SAMPLE_RATE)
                # Preprocess sample audio too
                ref_audio, ref_sr = preprocess_reference_audio(ref_audio, ref_sr)
                duration = get_audio_duration(ref_audio, ref_sr)
                if duration < 2:
                    raise HTTPException(
                        status_code=400,
                        detail="Sample audio is too short - minimum 2 seconds required"
                    )
            else:
                # Read uploaded file content (async)
                audio_bytes = await reference_audio.read()
                
                # Save to temporary file (librosa can auto-detect format from file)
                with tempfile.NamedTemporaryFile(suffix=Path(reference_audio.filename).suffix, delete=False) as tmp_upload:
                    tmp_upload.write(audio_bytes)
                    tmp_upload_path = tmp_upload.name
                    temp_audio_files.append(tmp_upload_path)  # Track for cleanup
                
                # Load with librosa (auto-detects format)
                import librosa
                try:
                    ref_audio, ref_sr = librosa.load(tmp_upload_path, sr=SAMPLE_RATE)
                except Exception as librosa_error:
                    # If librosa fails, use FFmpeg conversion
                    print(f"⚠️  Librosa load failed, attempting FFmpeg conversion...")
                    
                    try:
                        # Convert to WAV using FFmpeg
                        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        wav_path = wav_tmp.name
                        wav_tmp.close()
                        
                        wav_path = convert_to_wav_ffmpeg(tmp_upload_path, wav_path)
                        
                        # Now load the converted WAV file with librosa
                        ref_audio, ref_sr = librosa.load(wav_path, sr=SAMPLE_RATE)
                        
                        # Clean up converted WAV file
                        try:
                            os.unlink(wav_path)
                        except:
                            pass
                    except Exception as ffmpeg_error:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to process audio format: {str(ffmpeg_error)[:100]}"
                        )
                
                ref_audio, ref_sr = preprocess_reference_audio(ref_audio, ref_sr)
                duration = get_audio_duration(ref_audio, ref_sr)
                
                if duration < 2:
                    raise HTTPException(
                        status_code=400,
                        detail="Reference audio is too short - minimum 2 seconds required"
                    )
        except HTTPException:
            raise
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"❌ Error processing reference audio: {str(e)}")
            print(error_details)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process reference audio: {str(e)}"
            )
        
        # Save reference audio to temporary file for TTS processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_audio(ref_audio, tmp.name, ref_sr)
            ref_audio_path = tmp.name
            temp_audio_files.append(ref_audio_path)  # Track for cleanup
        
        try:
            # Generate output filename
            output_filename = f"synthesis_{hash(text)}.wav"
            output_path = OUTPUT_DIR / output_filename
            
            # Synthesize
            audio_data, sr, saved_path = engine.synthesize(
                text=text,
                reference_audio_path=ref_audio_path,
                language_code=language,
                output_path=str(output_path)
            )
            
            # Return audio file
            return FileResponse(
                path=saved_path,
                filename=output_filename,
                media_type="audio/wav"
            )
        
        finally:
            # Clean up all temporary files
            for temp_file in temp_audio_files:
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to clean up {temp_file}: {str(cleanup_error)}")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Synthesis error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}"
        )


@app.post("/synthesize-batch")
async def synthesize_batch(
    texts: str = Form(...),
    language: str = Form(default="en"),
    reference_audio: UploadFile = None
):
    """
    Batch synthesize speech (multiple texts separated by newlines)
    
    Parameters:
    - texts: Multiple texts separated by newlines
    - language: Language code (en=English, de=German)
    - reference_audio: Reference audio file for voice cloning
    
    Returns:
    - JSON with synthesis results for each text
    """
    
    if not reference_audio:
        raise HTTPException(status_code=400, detail="Please upload a reference audio file")
    
    if language not in LANGUAGE_OPTIONS:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    try:
        # Process reference audio
        ref_audio, ref_sr = bytesio_to_wav(reference_audio)
        ref_audio, ref_sr = preprocess_reference_audio(ref_audio, ref_sr)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_audio(ref_audio, tmp.name, ref_sr)
            ref_audio_path = tmp.name
        
        try:
            engine = get_tts_engine()
            if not engine.is_ready():
                raise HTTPException(status_code=503, detail="TTS model is loading")
            
            # Split texts by newline
            text_list = [t.strip() for t in texts.split("\n") if t.strip()]
            
            results = []
            for i, text in enumerate(text_list):
                try:
                    output_path = OUTPUT_DIR / f"batch_{i}_{hash(text)}.wav"
                    _, _, saved_path = engine.synthesize(
                        text=text,
                        reference_audio_path=ref_audio_path,
                        language_code=language,
                        output_path=str(output_path)
                    )
                    results.append({
                        "index": i,
                        "text": text,
                        "status": "success",
                        "filename": Path(saved_path).name
                    })
                except Exception as e:
                    results.append({
                        "index": i,
                        "text": text,
                        "status": "error",
                        "error": str(e)
                    })
            
            return JSONResponse({
                "total": len(text_list),
                "results": results
            })
        
        finally:
            try:
                os.unlink(ref_audio_path)
            except:
                pass
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use port 7860 for HuggingFace Spaces, default to 8000 otherwise
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
