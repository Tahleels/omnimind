import time
import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)
google_key = os.getenv("GOOGLE_API_KEY")

print("Waiting for Gemini rate limit reset...")
for i in range(1, 15):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={google_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": "Hello, respond with 'Gemini is working' if you see this."}]}]}
    r = requests.post(url, headers=headers, json=data)
    print(f"Attempt {i}: Status {r.status_code}")
    if r.status_code == 200:
        print("Success! Gemini response:")
        print(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        break
    else:
        err = r.json().get("error", {})
        msg = err.get("message", "")
        print(f"Error: {msg[:120]}")
    time.sleep(8)
