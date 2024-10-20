from ollama import Client, Options
import json
import requests
import aiohttp
from tempfile import NamedTemporaryFile
from PIL import Image
import io
import base64


class OllamaPromptGenerator:
    # Defaults
    OLLAMA_TIMEOUT = 90
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llava-llama3"  # Hardcoded for now
    OLLAMA_SYSTEM_MESSAGE = ("Use the supplied information to create a prompt for a next-generation "
                             "Natural Language Stable Diffusion model. Respond with only the final constructed prompt."
                             "Begin the prompt with the words: This is an (art or photography style here) image of "
                             )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("STRING", {"default": cls.OLLAMA_MODEL}),
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "system_message": ("STRING", {"default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "input_image": ("IMAGE",),  # Optional image input
                "clip": ("CLIP",),  # Optional CLIP input
                "unload_model": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    # All three outputs are defined
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning+", "conditioning-", "prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "Flux-O-llama"
    TITLE = "Ollama Flux Prompt Generator"

    def generate_prompt(self, ollama_url, ollama_model, text, system_message, seed=None, input_image=None,
                        clip=None, unload_model=False):
        """Generate a prompt using the Ollama API with optional multimodal inputs."""
        ollama_client = Client(host=ollama_url)

        opts = Options()
        if seed is not None and seed != 0:
            opts["seed"] = seed
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
        # Convert to a JSON string
        messages_string = json.dumps(messages)

        # Handle input image for multimodal models
        if input_image is not None:
            temp_image_path = OllamaHelpers.download_file(input_image.url)
            encoded_image = OllamaHelpers.resize_and_encode_image(temp_image_path)
            # Use the correct function to send the image
            response = ollama_client.generate(
                model=self.OLLAMA_MODEL,
                prompt=messages_string,
                images=[encoded_image],  # Wrap encoded image in a list
            )
            print(f"model response w image: {response}")

        else:
            # Call Ollama API to generate the prompt with no image
            response = ollama_client.generate(model=ollama_model, prompt=messages_string, options=opts)
            print(f"model response wo image: {response}")

        # Extract the prompt correctly
        imageprompt = response.get("response", "") + f" - initial tags: {text}"

        # Initialize conditioning outputs
        conditioning_positive = None  # Default empty for positive conditioning
        conditioning_negative = None
        if clip is not None:
            print("CLIP input provided, processing CLIP embeddings...")
            conditioning_positive = self.process_clip(clip, imageprompt)  # Pass the prompt for processing
            conditioning_negative = self.process_clip(clip, " ")  # Pass the prompt for processing

        # Unload model if option is selected (boolean Yes/No dropdown)
        if unload_model:
            print(f"Unloading model: {ollama_model}")
            OllamaHelpers.unload_model(ollama_url, ollama_model)

        # Return conditioning+ (with clip processing), conditioning- (empty string), and the prompt string
        return conditioning_positive, conditioning_negative, imageprompt  # Return the formatted prompt string instead of response

    @staticmethod
    def process_clip(clip, prompt):
        """Gets and encodes the prompt using CLIP."""
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]  # Return the appropriate structure


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
