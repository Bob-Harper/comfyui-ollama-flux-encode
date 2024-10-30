import requests
import torchvision.transforms as transforms
import io
import base64


class OllamaHelpers:
    @staticmethod
    def resize_and_encode_image(tensor_image):
        """Resize the image and encode it as base64."""
        # Convert tensor to PIL Image
        pil_image = transforms.ToPILImage()(tensor_image)

        # Resize the image to max 512x512
        max_size = 512
        pil_image.thumbnail((max_size, max_size))

        # Convert image to bytes and encode to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()

        return encoded_image

    @staticmethod    # Function to fetch models from the Ollama API
    def get_ollama_models(ollama_url="http://localhost:11434"):
        """Query the Ollama API to get the list of available models."""
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            models_data = response.json()  # The raw API response is a dictionary
            models = models_data.get('models', [])  # Access the 'models' key which contains the list of models

            # Extract 'name' from each model dictionary
            return [model['name'] for model in models]
        except requests.RequestException as e:
            print(f"Error fetching Ollama models: {e}")
            return []

    @staticmethod
    def unload_model(ollama_url, model_name):
        """Unload the model."""
        try:
            response = requests.post(f"{ollama_url}/api/generate", json={
                "model": model_name,
                "keep_alive": 0
            })
            response.raise_for_status()
            print(f"Model {model_name} unloaded successfully.")
        except requests.RequestException as e:
            print(f"Error unloading model {model_name}: {e}")


# At startup, we fetch the available models and store them
_available_models = OllamaHelpers.get_ollama_models()


# Function to return the cached model list
def available_models():
    return _available_models
