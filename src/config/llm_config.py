import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# Together.ai config
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

def get_together_headers():
    if not TOGETHER_API_KEY:
        raise EnvironmentError("TOGETHER_API_KEY not found in .env")
    return {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

def get_llm():
    return LLM(
        model="together_ai/mistralai/Mistral-7B-Instruct-v0.1",
        api_key=TOGETHER_API_KEY,
        temperature=0.0,
        verbose=True
    )