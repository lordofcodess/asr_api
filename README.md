# ASR API with Local Model Support

This is a FastAPI-based Automatic Speech Recognition (ASR) API that supports local Whisper models. It's designed to be easily adaptable for different languages and model configurations.

## Features

- FastAPI backend with automatic API documentation
- Support for local Whisper models
- Audio file conversion and preprocessing
- CORS enabled for cross-origin requests
- Simple and clean API interface

## Prerequisites

- Python 3.9 or higher
- FFmpeg installed on your system
- A local Whisper model (compatible with Hugging Face transformers)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setting Up Your Local Model

1. Place your local model in a directory named after your model (e.g., `whisper-small_Akan_non_standardspeech/`)
2. Ensure your model directory contains all necessary files:
   - config.json
   - model.safetensors (or model.bin)
   - tokenizer.json
   - preprocessor_config.json
   - special_tokens_map.json

3. Update the `MODEL_PATH` in `main.py` to point to your model:
```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), "your_model_directory")
```

## Running the API

1. Start the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. Access the API documentation at:
```
http://localhost:8000/docs
```

## API Endpoints

### POST /transcribe
Transcribes audio files to text.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - audio: Audio file (supported formats: wav, mp3, m4a, etc.)

**Response:**
```json
{
    "text": "transcribed text"
}
```

## Using with Mobile Apps

To use this API with a mobile app:

1. Make sure your computer and mobile device are on the same network
2. Find your computer's local IP address:
   - Windows: `ipconfig`
   - Mac/Linux: `ifconfig` or `ip addr`
3. Use the IP address in your API calls:
```javascript
const API_URL = 'http://YOUR_LOCAL_IP:8000/transcribe';
```

## Error Handling

The API returns appropriate error messages in case of:
- Invalid audio files
- Model loading issues
- Processing errors

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license] 