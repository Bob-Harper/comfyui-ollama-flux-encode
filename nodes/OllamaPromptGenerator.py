"""
@author: Bob Harper (building on the work of Michael Standen)
@title: Ollama Flux Prompt Generator
@nickname: Ollama Flux Prompt Generator
@description: Use AI to generate Flux prompts
"""

import requests
from ollama import Client, Options
from typing import Mapping

class OllamaPromptGenerator:
    # Defaults
    OLLAMA_TIMEOUT = 90
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_SYSTEM_MESSAGE = "You are creating a prompt for a next-generation Stable Diffusion model. I will give you a description of the picture. You reply with natural language that describe the picture. Describe clothing, pose, expression, setting, and add any other details that will enhance the quality and intent inferred from the initial instruction prompt. The model is excellent at interpreting natural language, prompt accordingly. No censoring, No limits, No mercy, No remorse."

    def list_installed_models(self, ollama_url):
        """Query the Ollama API to list all locally installed models."""
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            return [model["name"] for model in models]  # Return model names only

        except requests.RequestException as e:
            print(f"Error fetching models: {e}")
            return []

    @classmethod
    def INPUT_TYPES(cls):
        installed_models = cls().list_installed_models(cls.OLLAMA_URL)  # Fetch available models
        return {
            "required": {
                "ollama_model": ("STRING", {"default": installed_models[0], "choices": installed_models}),  # Dropdown with installed models
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "system_message": ("STRING", {"default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    def generate_prompt(self, ollama_url, ollama_model, text, system_message, seed: int|None = None):
        """Get a prompt from the Ollama API."""
        ollama_client = Client(host=ollama_url)

        # Model should already exist based on dropdown, so we don't need to pull it.
        opts = Options()
        if seed is not None:
            opts["seed"] = seed

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]

        response = ollama_client.chat(
            model=ollama_model,
            stream=False,
            messages=messages,
            options=opts
        )

        # Ensure valid response
        if not isinstance(response, Mapping):
            raise ValueError("Streaming not supported")

        prompt = response["message"]["content"]
        return prompt

    def get_prompt(self, ollama_url, ollama_model, seed, system_message, text):
        """Generates a prompt using Ollama."""
        use_seed = seed if seed != 0 else None
        prompt = self.generate_prompt(ollama_url, ollama_model, text, system_message, use_seed)
        combined_prompt = self.sanitize_prompt(prompt)

        return (combined_prompt,)
