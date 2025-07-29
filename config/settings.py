# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from a .env file

# Groq API key setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Vector store configurations (for FAISS, Pinecone, etc.)
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")  # You can switch to pinecone or chroma later
