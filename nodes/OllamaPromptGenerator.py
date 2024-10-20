"""
@author: Bob Harper (building on the work of Michael Standen)
@title: Ollama Flux Prompt Generator
@nickname: Ollama Flux Prompt Generator
@description: Use AI to generate Flux prompts
"""

from ollama import Client, Options
import requests


class OllamaPromptGenerator:
    # Defaults
    OLLAMA_TIMEOUT = 90
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "Llama3.2"  # Hardcoded for now
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
                "vae": ("VAE",),  # Optional VAE input
                "clip": ("CLIP",),  # Optional CLIP input
                "unload_model": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preview": ("STRING", {"default": "", "multiline": True, "placeholder": "Preview"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("conditioning", "prompt", "image", "latent", "preview")
    FUNCTION = "generate_prompt"
    CATEGORY = "Flux-O-llama"
    TITLE = "Ollama Flux Prompt Generator"

    @staticmethod
    def generate_prompt(ollama_url, ollama_model, text, system_message, seed=None, input_image=None, vae=None,
                        clip=None, latent=None, conditioning=None, unload_model=False):
        """Generate a prompt using the Ollama API with optional multimodal inputs."""
        ollama_client = Client(host=ollama_url)

        opts = Options()
        if seed is not None and seed != 0:
            opts["seed"] = seed

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]

        # Optional: Handle input image for multimodal models
        if input_image is not None:
            messages.append({"role": "user", "content": "Input image supplied for multimodal processing."})
            print("Image supplied, using multimodal model.")

        # Call Ollama API to generate the prompt
        response = ollama_client.chat(model=ollama_model, messages=messages, options=opts)
        prompt = response["message"]["content"] + f" - initial tags: {text}"

        # Log the full response
        print("Full Response to send to the model:", prompt)

        # Placeholder for CLIP processing
        if clip is not None:
            print("CLIP input provided, processing CLIP embeddings...")
            conditioning = OllamaPromptGenerator.process_clip(clip)

        # Placeholder for VAE processing
        if vae is not None and input_image is not None:
            print("VAE provided, encoding image...")
            input_image = OllamaPromptGenerator.process_vae(vae, input_image)

        # Placeholder for latent input processing
        if latent is not None:
            print("Latent input provided, processing latent...")
            latent = OllamaPromptGenerator.process_latent(latent)

        # Unload model if option is selected (boolean Yes/No dropdown)
        if unload_model:
            print(f"Unloading model: {ollama_model}")
            OllamaPromptGenerator.unload_model(ollama_url, ollama_model)

        # Return conditioning, prompt, image, and latent values
        return conditioning, prompt, input_image, latent, {"ui": {"text": prompt}, "result": (prompt,)}

    @staticmethod
    def process_clip(clip):
        """Placeholder function for CLIP processing."""
        # Add CLIP processing logic here
        return clip

    @staticmethod
    def process_vae(vae, input_image):
        """Placeholder function for VAE processing."""
        # Add VAE image encoding logic here
        return input_image

    @staticmethod
    def process_latent(latent):
        """Placeholder function for latent input processing."""
        # Add latent processing logic here
        return latent

    @staticmethod
    def unload_model(ollama_url, model_name):
        """Unload the specified model to free up memory."""
        try:
            response = requests.post(f"{ollama_url}/api/generate", json={
                "model": model_name,
                "keep_alive": 0
            })
            response.raise_for_status()
            print(f"Model {model_name} unloaded successfully.")
        except requests.RequestException as e:
            print(f"Error unloading model {model_name}: {e}")
