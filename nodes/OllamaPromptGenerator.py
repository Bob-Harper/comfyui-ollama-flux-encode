from ollama import Client, Options
import json
import requests
from PIL import Image
import torchvision.transforms as transforms
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
        messages_string = json.dumps(messages)

        # Handle the single input image for multimodal models
        # Handle the single input image for multimodal models
        if input_image is not None:
            # Ensure correct dimensions: 4D tensor [1, channels, height, width]
            if input_image.ndimension() == 4:
                print(f"Original Image dimensions: {input_image.size()}")  # Log the image dimensions

                # Remove the batch dimension and keep only the first 3 channels (RGB)
                input_image = input_image[0, :3, :, :]  # Remove batch and select RGB channels

                print(
                    f"Processed Image dimensions (after batch removal): {input_image.size()}")  # Log the processed dimensions

                # Convert the tensor image to a PIL Image and resize/encode
                try:
                    encoded_image = OllamaHelpers.resize_and_encode_image(input_image)
                    print(f"Encoded image: {encoded_image[:30]}...")

                    # Make API request with the image
                    response = ollama_client.generate(
                        model=ollama_model,
                        prompt=messages_string,
                        images=[encoded_image],  # Pass the encoded image
                    )
                except Exception as e:
                    print(f"Error processing image: {e}")
                    response = {"response": ""}  # Handle any failure gracefully
            else:
                print(f"Unexpected image dimensions: {input_image.ndimension()} dimensions found.")
                response = {"response": ""}

        else:
            # Call Ollama API to generate the prompt without an image
            response = ollama_client.generate(model=ollama_model, prompt=messages_string, options=opts)
            print(f"Model response without image: {response}")

        # Extract the prompt from the response
        imageprompt = response.get("response", "") + f" - initial tags: {text}"

        # Handle CLIP conditioning if provided
        conditioning_positive = None
        conditioning_negative = None
        if clip is not None:
            conditioning_positive = self.process_clip(clip, imageprompt)  # Pass the prompt to CLIP
            conditioning_negative = self.process_clip(clip, " ")  # Send an empty prompt for negative conditioning

        # Unload the model if the option is selected
        if unload_model:
            OllamaHelpers.unload_model(ollama_url, ollama_model)

        # Return positive conditioning, negative conditioning, and the prompt
        return conditioning_positive, conditioning_negative, imageprompt

    @staticmethod
    def process_clip(clip, prompt):
        """Process the prompt with CLIP."""
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]  # Return CLIP conditioning


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
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return encoded_image

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
