import requests
import aiohttp
from tempfile import NamedTemporaryFile
from PIL import Image
import io
import base64


class OllamaHelpers:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def get_loaded_models(self):
        """Check for loaded models."""
        try:
            response = requests.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error retrieving loaded models: {e}")
            return None

    def unload_model(self, model_name):
        """Unload the specified model."""
        try:
            response = requests.post(f"{self.base_url}/api/generate", json={
                "model": model_name,
                "keep_alive": 0
            })
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error unloading model {model_name}: {e}")
            return None

    @staticmethod
    async def download_file(url: str) -> str:
        """Download a file from a URL and return the path to the downloaded file."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    # Create a temporary file
                    with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                        temp_file.write(await resp.read())
                        return temp_file.name
                else:
                    raise FileNotFoundError(f"Failed to download file from {url}. Status code: {resp.status}")

    @staticmethod
    def resize_and_encode_image(ximage_path, max_size=(512, 512)):
        with Image.open(ximage_path) as img:
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.read()).decode()
            return encoded_image

# Example usage:
# ollama_helper = OllamaHelpers()
# loaded_models = ollama_helper.get_loaded_models()
# ollama_helper.unload_model("Llama3.2")
