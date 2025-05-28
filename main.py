from fastapi import FastAPI, UploadFile, File, Response, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import os
from typing import Optional
import torch
from transformers import pipeline, AutoProcessor, AutoModel
from pathlib import Path
from pydub import AudioSegment
import tempfile
import noisereduce as nr
import soundfile as sf
import io
import base64
import requests
from faster_whisper import WhisperModel
from TTS.api import TTS
import asyncio
from contextlib import asynccontextmanager

# Initialize global variables
tts_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
TTS_MODEL_NAME = "tts_models/tw_asante/openbible/vits"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tts_model
    try:
        print(f"Loading TTS model on {device}...")
        tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False).to(device)
        print("TTS Model loaded successfully!")
    except Exception as e:
        print(f"TTS Model loading failed: {str(e)}")
        tts_model = None
    
    yield
    
    # Shutdown
    if tts_model is not None:
        del tts_model
        torch.cuda.empty_cache()

app = FastAPI(
    title="Speech API",
    description="API for Automatic Speech Recognition and Text-to-Speech",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ASR model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "faster_whisper_akan")

# Load ASR model at startup
asr_model = WhisperModel(MODEL_PATH, device=device, compute_type="float16" if device == "cuda" else "float32")

def convert_to_wav(audio_file_path: str) -> str:
    """Convert any audio file to WAV format with 16kHz sample rate."""
    # Create a temporary file for the WAV output
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    
    # Load and convert the audio
    audio = AudioSegment.from_file(audio_file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(temp_wav.name, format="wav")
    
    return temp_wav.name

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Faster Whisper."""
    # Read the WAV file
    audio_np, sample_rate = sf.read(audio_path)
    
    # Ensure audio is float32 and normalized
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    if np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np / 32768.0
    
    # Apply noise reduction
    audio_np = nr.reduce_noise(
        y=audio_np,
        sr=sample_rate,
        prop_decrease=0.7,  # Amount of noise to reduce (0-1)
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50
    )
    
    # Run inference with Faster Whisper
    segments, _ = asr_model.transcribe(
        audio_np,
        language="yo",  # Akan language code
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Combine all segments into a single text
    transcription = " ".join([segment.text for segment in segments])
    return transcription

@app.get("/")
async def root():
    return {"message": "Welcome to Speech API"}

@app.post("/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...)
):
    """
    Transcribe audio file to text using the ASR model
    """
    try:
        # Save the uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        temp_input.close()
        
        with open(temp_input.name, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # Convert to WAV format
        wav_path = convert_to_wav(temp_input.name)
        
        # Transcribe the audio
        transcription = transcribe_audio(wav_path)
        
        # Clean up temporary files
        os.remove(temp_input.name)
        os.remove(wav_path)
        
        return {
            "text": transcription
        }
    
    except Exception as e:
        return {
            "error": str(e)
        }

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Text to synthesize")
    speaker_id: Optional[str] = Field(None, description="Optional speaker ID for multi-speaker models")
    language: Optional[str] = Field("tw", description="Language code (default: tw for Asante)")

@app.post("/synthesize")
async def synthesize_speech_post(request: TTSRequest):
    """
    Convert text to speech using TTS model (POST method)
    
    Args:
        request: TTSRequest object containing:
            - text: Text to synthesize (required)
            - speaker_id: Optional speaker ID
            - language: Optional language code (default: tw)
    """
    if not tts_model:
        raise HTTPException(
            status_code=503,
            detail="TTS model is not loaded. Please try again in a few moments."
        )
    
    try:
        # Create in-memory audio buffer
        audio_buffer = io.BytesIO()
        
        # Generate speech
        kwargs = {
            "language": request.language
        }
        if request.speaker_id:
            kwargs["speaker"] = request.speaker_id
            
        tts_model.tts_to_file(
            text=request.text,
            file_path=audio_buffer,
            **kwargs
        )
        
        # Return as streaming response
        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Generated-Text": request.text[:100],  # Return first 100 chars for reference
                "Content-Type": "audio/wav"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}"
        )

@app.get("/synthesize")
async def synthesize_speech_get(
    text: str = Field(..., min_length=1, max_length=500),
    speaker_id: Optional[str] = None,
    language: Optional[str] = "tw"
):
    """
    Convert text to speech using TTS model (GET method)
    
    Args:
        text: Text to synthesize (required)
        speaker_id: Optional speaker ID
        language: Optional language code (default: tw)
    """
    if not tts_model:
        raise HTTPException(
            status_code=503,
            detail="TTS model is not loaded. Please try again in a few moments."
        )
    
    try:
        # Create in-memory audio buffer
        audio_buffer = io.BytesIO()
        
        # Generate speech
        kwargs = {
            "language": language
        }
        if speaker_id:
            kwargs["speaker"] = speaker_id
            
        tts_model.tts_to_file(
            text=text,
            file_path=audio_buffer,
            **kwargs
        )
        
        # Return as streaming response
        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Generated-Text": text[:100],  # Return first 100 chars for reference
                "Content-Type": "audio/wav"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "ready" if tts_model else "loading",
        "device": device,
        "asr_model": "faster_whisper_akan",
        "tts_model": TTS_MODEL_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 