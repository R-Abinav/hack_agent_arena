from appworld import AppWorld
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key, transport='rest')

try:
    with AppWorld(task_id='50e1ac9_1') as world:
        model = genai.GenerativeModel("gemini-flash-latest")
        print("Generating content inside AppWorld context with REST transport...")
        response = model.generate_content("Hello")
        print("Response:", response.text)
except Exception as e:
    print("Error:", e)
