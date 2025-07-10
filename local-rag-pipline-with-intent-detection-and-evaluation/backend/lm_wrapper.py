import openai
import requests
import json

LM_STUDIO_URL = "http://localhost:5000/v1/completions"

class LMWrapper:
    def __init__(self, use_local=True):
        self.use_local = use_local
        self.local_headers = {'Content-Type': 'application/json'}
        self.local_model = 'qwen3-4b'  # Name of the local model
        self.openai_api_key = 'your-openai-api-key'
    
    def query_local_model(self, prompt):
        try:
            client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            response = client.chat.completions.create(
                model=self.local_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Local model error: {e}")
            return f"I understand your query about: {prompt}. This is a simulated response as the local model is not available."

    def query_openai(self, prompt):
        try:
            openai.api_key = self.openai_api_key
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            return f"I understand your query about: {prompt}. This is a simulated response as OpenAI is not configured."
    
    def query(self, prompt):
        if self.use_local:
            return self.query_local_model(prompt)
        else:
            return self.query_openai(prompt) 