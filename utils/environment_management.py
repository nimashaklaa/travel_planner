import os
from dotenv import load_dotenv

load_dotenv()

def load_api_key():
    return os.getenv("OPENAI_API_KEY")

def load_database_url():
    return os.getenv("DATABASE_URL")
