# ASR (Automatic Speech Recognition) API

This API provides automatic speech recognition capabilities using the Whisper model, with additional features like noise reduction and audio format conversion.

## Features

- Audio transcription using Whisper model
- Noise reduction processing
- Support for various audio formats (automatically converts to WAV)
- CORS enabled for cross-origin requests
- GPU acceleration support (if available)

## Dependencies

```bash
pip install fastapi
pip install uvicorn
pip install numpy
pip install torch
pip install transformers
pip install pydub
pip install noisereduce
pip install soundfile
```

## Code Structure

### Imports and Setup

```python
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
import noisereduce as nr
```

- `FastAPI`: Web framework for building APIs
- `CORSMiddleware`: Enables Cross-Origin Resource Sharing
- `numpy`: For numerical operations on audio data
- `torch`: For GPU acceleration
- `transformers`: For the Whisper ASR model
- `pydub`: For audio file conversion
- `noisereduce`: For noise reduction processing

### API Configuration

```python
app = FastAPI(title="ASR API", description="API for Automatic Speech Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- Creates FastAPI application
- Configures CORS to allow requests from any origin

### Model Initialization

```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), "whisper-small_Akan_non_standardspeech")
device = "cuda" if torch.cuda.is_available() else "cpu"

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,
    device=device
)
```

- Loads the Whisper model from local path
- Uses GPU if available, falls back to CPU
- Initializes the ASR pipeline

### Audio Conversion Function

```python
def convert_to_wav(audio_file_path: str) -> str:
    """Convert any audio file to WAV format with 16kHz sample rate."""
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    
    audio = AudioSegment.from_file(audio_file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(temp_wav.name, format="wav")
    
    return temp_wav.name
```

- Converts any audio format to WAV
- Sets sample rate to 16kHz
- Converts to mono channel
- Returns path to temporary WAV file

### Transcription Function

```python
def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using the Hugging Face pipeline."""
    import soundfile as sf
    audio_np, sample_rate = sf.read(audio_path)
    
    # Normalize audio
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    if np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np / 32768.0
    
    # Noise reduction
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
    
    # Prepare and process audio
    audio_input = {
        "raw": audio_np,
        "sampling_rate": 16000
    }
    
    result = asr_pipe(audio_input, generate_kwargs={"language": "yo"})
    return result["text"]
```

- Reads audio file using soundfile
- Normalizes audio to float32 format
- Applies noise reduction with the following parameters:
  - `prop_decrease=0.7`: Reduces 70% of the noise
  - `n_fft=2048`: FFT window size for spectral analysis
  - `win_length=2048`: Window length for STFT
  - `hop_length=512`: Samples between windows
  - `time_constant_s=2.0`: Time constant for noise estimation
  - `freq_mask_smooth_hz=500`: Frequency smoothing
  - `time_mask_smooth_ms=50`: Time smoothing
- Processes audio through ASR pipeline
- Returns transcribed text

### API Endpoints

#### Root Endpoint
```python
@app.get("/")
async def root():
    return {"message": "Welcome to ASR API"}
```
- Simple welcome message

#### Transcription Endpoint
```python
@app.post("/transcribe")
async def transcribe_audio_endpoint(audio: UploadFile = File(...))
```
- Accepts audio file upload
- Converts to WAV format
- Applies noise reduction
- Returns transcription

### Server Configuration

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

- Runs the API server on localhost:8000

## Usage

1. Start the server:
```bash
python main.py
```

2. Send a POST request to `http://127.0.0.1:8000/transcribe` with an audio file
3. Receive the transcription in the response

## Error Handling

The API includes error handling for:
- File upload issues
- Audio conversion problems
- Transcription errors
- Temporary file cleanup

## Notes

- The API uses temporary files for processing
- All temporary files are automatically cleaned up
- The model is configured for Yoruba language ("yo")
- GPU acceleration is automatically enabled if available 