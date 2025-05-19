import requests

def test_transcribe(audio_file_path):
    # API endpoint
    url = "http://localhost:8000/transcribe"
    
    # Prepare the file
    files = {
        'file': ('audio.wav', open(audio_file_path, 'rb'), 'audio/wav')
    }
    
    # Make the request
    response = requests.post(url, files=files)
    
    # Print the result
    print("Status:", response.status_code)
    print("Transcription:", response.json()['transcription'])

# Example usage
if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "path/to/your/audio.wav"
    test_transcribe(audio_file) 