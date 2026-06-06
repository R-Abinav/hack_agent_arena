import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("gemini-flash-latest")
    print("Generating content...")
    response = model.generate_content("Hello")
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
