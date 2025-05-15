from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from typing import Optional
import torch
from transformers import pipeline
from pathlib import Path
from pydub import AudioSegment
import tempfile

app = FastAPI(title="ASR API", description="API for Automatic Speech Recognition")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ASR model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "whisper-small_Akan_non_standardspeech")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ASR pipeline at startup
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,
    device=device
)

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
    """Transcribe audio using the Hugging Face pipeline."""
    # Read the WAV file
    import soundfile as sf
    audio_np, _ = sf.read(audio_path)
    
    # Ensure audio is float32 and normalized
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    if np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np / 32768.0
    
    # Prepare audio input
    audio_input = {
        "raw": audio_np,
        "sampling_rate": 16000
    }
    
    # Run inference
    result = asr_pipe(audio_input, generate_kwargs={"language": "yo"})
    return result["text"]

@app.get("/")
async def root():
    return {"message": "Welcome to ASR API"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="YOUR_LOCAL_IP_ADDRESS", port=8000) 