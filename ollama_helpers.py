import base64
import io
from PIL import Image
import requests
import numpy as np


class OllamaHelpers:
    ollama_url = "http://localhost:11434"  # Default URL for the Ollama API

    @staticmethod
    def resize_and_encode_image(input_image):
        """Resize tensor image to 512x512 and encode as base64."""

        # Ensure the tensor has the correct shape
        if input_image.ndimension() == 4:
            input_image = input_image[0]  # Remove batch dimension

        # Convert tensor to numpy array and check shape and type
        image_np = (255.0 * input_image.cpu().numpy()).clip(0, 255).astype(np.uint8)

        # Ensure shape is [height, width, channels] for RGB
        if image_np.shape[0] == 3:  # Check if channels are first
            image_np = np.transpose(image_np, (1, 2, 0))  # Rearrange to [H, W, C]
        elif image_np.shape[-1] != 3:
            raise ValueError(f"Unexpected shape for RGB image: {image_np.shape}")

        # Create PIL image and resize
        pil_image = Image.fromarray(image_np)
        pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)

        # Encode image as base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print(f"Encoded image length: {len(encoded_image)} characters")
        return encoded_image

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
print(f"\033[95mAvailable Ollama Models: {_available_models}\033[0m")
