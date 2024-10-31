import base64
import io

import requests
import torchvision.transforms as transforms


class OllamaHelpers:
    ollama_url = "http://localhost:11434"  # Default URL for the Ollama API

    @staticmethod
    def resize_and_encode_image(tensor_image):
        """Resize the image and encode it as base64."""
        pil_image = transforms.ToPILImage()(tensor_image)
        pil_image.thumbnail((512, 512))  # Resize to max 512x512

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    @classmethod
    def get_available_models(cls):
        """Query the Ollama API to get the list of available models."""
        try:
            response = requests.get(f"{cls.ollama_url}/api/tags")
            response.raise_for_status()
            models_data = response.json()
            return [model['name'] for model in models_data.get('models', [])]  # Directly extract model names
        except requests.RequestException as e:
            print(f"Error fetching Ollama models: {e}")
            return []

    @classmethod
    def unload_model(cls, model_name):
        """Unload the model."""
        try:
            response = requests.post(f"{cls.ollama_url}/api/generate", json={
                "model": model_name,
                "keep_alive": 0
            })
            response.raise_for_status()
            print(f"Model {model_name} unloaded successfully.")
        except requests.RequestException as e:
            print(f"Error unloading model {model_name}: {e}")


# Fetch available models at startup
_available_models = OllamaHelpers.get_available_models()
print(f"Available Ollama Models: ", _available_models)
