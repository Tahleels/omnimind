import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

google_key = os.getenv("GOOGLE_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

print("--- Testing Google Gemini API ---")
try:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={google_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": "Hello, respond with 'Gemini is working' if you see this."}]}]}
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:", response.json()["candidates"][0]["content"]["parts"][0]["text"])
    else:
        print("Error Response:", response.text)
except Exception as e:
    print("Google Gemini API error:", e)

print("\n--- Testing OpenRouter API ---")
try:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AskMyDocs test",
    }
    data = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": [{"role": "user", "content": "Hello, respond with 'OpenRouter is working' if you see this."}],
    }
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:", response.json()["choices"][0]["message"]["content"])
    else:
        print("Error Response:", response.text)
except Exception as e:
    print("OpenRouter API error:", e)
