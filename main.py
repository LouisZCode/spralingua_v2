#Minimax Call Testing
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MINIMAX_API_KEY")

text = "Servus! Ich bin deine Sprachlehrer aus Bayern. Gemeinsam lernen wir Deutsch auf eine entspannte und freundliche Art. Lass uns anfangen!"

response = requests.post(
    "https://api.minimax.io/v1/t2a_v2",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "speech-2.6-hd",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": "german_bavarian_male_v3",   #german_bavarian_female, german_bavarian_male_v2
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "emotion": "happy"
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1
        },
        "language_boost": "auto"
    }
)

# Get hex audio from correct location
audio_hex = response.json()["data"]["audio"]

# Convert hex to bytes
audio_bytes = bytes.fromhex(audio_hex)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)

print("Done! Audio saved to output.mp3")

