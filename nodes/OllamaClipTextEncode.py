"""
@author: Bob Harper (building on the work of Michael Standen)
@title: Ollama Flux Prompt Encode
@nickname: Ollama Flux Prompt Encode
@description: Use AI to generate Flux style prompts and perform CLIP text encoding
"""
import requests

from .OllamaPromptGenerator import OllamaPromptGenerator


class OllamaCLIPTextEncode(OllamaPromptGenerator):
    
    @classmethod
    def INPUT_TYPES(cls):
        # Fetch the model list for the dropdown
        model_list = cls.fetch_model_list()

        return {
            "required": {
                "clip": ("CLIP",),
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "ollama_model": ("COMBO", model_list),  # Use COMBO for dropdown
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prepend_tags": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    @classmethod
    def fetch_model_list(cls):
        """Fetch the available models from the Ollama API."""
        try:
            response = requests.get(f"{cls.OLLAMA_URL}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            # Extract model names for the dropdown
            return [model["name"] for model in models]
        except requests.RequestException as e:
            print(f"Error fetching models: {e}")
            return ["Model not available"]  # Fallback option

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = (
        "conditioning",
        "prompt",
    )
    FUNCTION = "get_encoded"

    CATEGORY = "Ollama"

    def unload_model(self, ollama_url, model_name):
        """Unload the specified model."""
        try:
            response = requests.post(f"{ollama_url}/api/generate", json={
                "model": model_name,
                "keep_alive": 0
            })
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error unloading model {model_name}: {e}")
            return None

    def get_encoded(self, clip, ollama_url, ollama_model, seed, prepend_tags, text):
        # Unload the model before encoding
        unload_response = self.unload_model(ollama_url, ollama_model)
        if unload_response and unload_response.get("done"):
            print(f"Successfully unloaded model: {unload_response['model']}")
        else:
            print("Failed to unload the model.")
        """Gets and encodes the prompt using CLIP."""
        # Fetch and sanitize the prompt using inherited method
        combined_prompt = self.get_prompt(ollama_url, ollama_model, seed, prepend_tags, text)[0]

        # Tokenize and encode the prompt with CLIP
        tokens = clip.tokenize(combined_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return [[cond, {"pooled_output": pooled}]], combined_prompt
