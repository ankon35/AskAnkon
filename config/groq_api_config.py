# config/groq_api_config.py
import requests
import os

class GroqAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com"  # Assuming the Groq API URL is something like this
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_chat_completion(self, messages, model="llama-3.3-70b-versatile"):
        """Fetches chat completions from Groq's API."""
        url = f"{self.base_url}/openai/v1/chat/completions"
        data = {
            "model": model,
            "messages": messages
        }
        response = requests.post(url, headers=self.headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            raise Exception(f"Failed to fetch chat completion: {response.status_code}, {response.text}")
    
    def get_embedding(self, text):
        """Fetches embeddings from Groq's API using chat completion as a workaround."""
        # Since Groq doesn't have a direct embeddings endpoint, we'll use chat completion
        # to generate a response that can be used for similarity
        messages = [{"role": "user", "content": f"Generate a concise summary of: {text}"}]
        return self.get_chat_completion(messages)
