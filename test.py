import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

url = (
    "https://ai.api.nvidia.com/v1/genai/"
    "black-forest-labs/flux-3-image-klein"
)

payload = {
    "prompt": "A simple red apple on a wooden table",
    "width": 512,
    "height": 512,
    "steps": 20
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=120
)

data = response.json()

if response.status_code == 200:
    image_b64 = data["artifacts"][0]["base64"]

    with open("apple.jpg", "wb") as f:
        f.write(base64.b64decode(image_b64))

    print("✅ Saved image: apple.jpg")

else:
    print("❌ Error")
    print(response.status_code)
    print(data)
