from ollama import Client, Options
import requests
from ollama_helpers import OllamaHelpers
import json

ollama_helper = OllamaHelpers()


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

        # Initialize prompt to ensure it's always defined
        prompt = ""  # Default initialization

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
        # Convert to a JSON string
        messages_string = json.dumps(messages)
        # Handle input image for multimodal models
        if input_image is not None:
            try:
                temp_image_path = ollama_helper.download_file(input_image.url)
                encoded_image = ollama_helper.resize_and_encode_image(temp_image_path)
                # Use the correct function to send the image
                chat_response = ollama_client.generate(
                    model=self.OLLAMA_MODEL,
                    prompt=messages_string,
                    images=[encoded_image],  # Wrap encoded image in a list
                )

                print(f"chat_response: {chat_response}")

                # Check the response and extract content
                if "response" in chat_response:
                    prompt = chat_response["response"]
                else:
                    print("Ow my head..  who slipped what into my drink.....")

            except Exception as e:
                print(f"Something went wrong. I DO NOT LIKE BEING WRONG: {e}")

        else:
            # Call Ollama API to generate the prompt
            response = ollama_client.chat(model=ollama_model, messages=messages, options=opts)
            prompt = response["message"]["content"] + f" - initial tags: {text}"

        # Initialize conditioning outputs
        conditioning_positive = None  # Default empty for positive conditioning
        conditioning_negative = None
        if clip is not None:
            print("CLIP input provided, processing CLIP embeddings...")
            conditioning_positive = self.process_clip(clip, prompt)  # Pass the prompt for processing
            conditioning_negative = self.process_clip(clip, " ")  # Pass the prompt for processing
        # Unload model if option is selected (boolean Yes/No dropdown)
        if unload_model:
            print(f"Unloading model: {ollama_model}")
            OllamaPromptGenerator.unload_model(ollama_url, ollama_model)

        # Return conditioning+ (with clip processing), conditioning- (empty string), and prompt
        return conditioning_positive, conditioning_negative, prompt  # Return prompt for further processing

    @staticmethod
    def process_clip(clip, prompt):
        """Gets and encodes the prompt using CLIP."""
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]  # Return the appropriate structure

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
