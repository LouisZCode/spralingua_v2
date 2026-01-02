#Minimax Call Testing
import requests
import os
from dotenv import load_dotenv

import requests
import os
from dotenv import load_dotenv

load_dotenv()


#agent testing
from agents import test_agent

test_message = "Dime un chiste sobre san luis potosi"

result = test_agent.invoke(
    {"messages":[{"role": "user", "content": test_message}]}
)

answer = result["messages"][-1].content

api_key = os.getenv("MINIMAX_API_KEY")

text = answer

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
            "voice_id": "female-shaonv",
            "speed": 1,
            "vol": 1,
            "pitch": 0
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1
        }
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




#agent testing
from agents import test_agent

test_message = "Dime un chiste sobre san luis potosi"

result = test_agent.invoke(
    {"messages":[{"role": "user", "content": test_message}]}
)

for i, msg in enumerate(result["messages"]):
    msg.pretty_print() 