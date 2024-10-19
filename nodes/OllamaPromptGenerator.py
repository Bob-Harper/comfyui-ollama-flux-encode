"""
@author: Bob Harper (building on the work of Michael Standen)
@title: Ollama Flux Prompt Generator
@nickname: Ollama Flux Prompt Generator
@description: Use AI to generate Flux prompts
"""

from ollama import Client, Options


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
                "ollama_model": ("STRING", {"default": cls.OLLAMA_MODEL}),  # Hardcoded model field
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "system_message": ("STRING", {"default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "FluxOllama"
    TITLE = "Ollama Flux Prompt Generator"

    @staticmethod
    def generate_prompt(ollama_url, ollama_model, text, system_message, seed: int | None = None):
        """Get a prompt from the Ollama API."""
        ollama_client = Client(host=ollama_url)

        opts = Options()
        if seed is not None:
            opts["seed"] = seed

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]

        response = ollama_client.chat(model=ollama_model, messages=messages, options=opts)
        prompt = response["message"]["content"] + f" - Also include initial tags describing the image: {text}"
        print("Full Response to send to the model:", prompt)  # Log the full response
        return (prompt,)  # THIS NEEDS THE COMMA AFTER PROMPT.  don't ask why.  accept it.
