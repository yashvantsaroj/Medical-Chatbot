import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
else:
    try:
        genai.configure(api_key=api_key)
        print("Checking available models...")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                available_models.append(m.name)
        
        if not available_models:
            print("No models found that support 'generateContent'. Check your API key and region availability.")
    except Exception as e:
        print(f"An error occurred: {e}")
