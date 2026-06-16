import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

SYSTEM_PROMPT = "You are a test agent."
prompt = f"{SYSTEM_PROMPT}\n\nPast lessons: None."

messages = [
    {"role": "user", "content": "Supervisor: admin\n\nTask: Test task.\n\nBegin."}
]

gemini_messages = []
for msg in messages:
    role = "model" if msg["role"] == "assistant" else "user"
    gemini_messages.append({"role": role, "parts": [msg["content"]]})

gemini_messages[0]["parts"][0] = f"{prompt}\n\n{gemini_messages[0]['parts'][0]}"

model = genai.GenerativeModel("gemini-flash-latest")

print("Sending request...")
try:
    resp = model.generate_content(
        contents=gemini_messages,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
        ),
    )
    print("Response:", resp.text)
except Exception as e:
    print("Error:", e)
