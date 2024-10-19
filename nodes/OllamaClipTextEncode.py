"""
@author: Bob Harper (building on the work of Michael Standen)
@title: Ollama Flux Prompt Encode
@nickname: Ollama Flux Prompt Encode
@description: Use AI to generate Flux style prompts and perform CLIP text encoding
"""
from .OllamaPromptGenerator import OllamaPromptGenerator

class OllamaCLIPTextEncode(OllamaPromptGenerator):

    @classmethod
    def INPUT_TYPES(cls):
        # Fetch available models when the node is initialized, not at class definition time.
        try:
            installed_models = cls.list_installed_models(cls.OLLAMA_URL)  # Fetch available models
        except Exception as e:
            print(f"Error fetching models: {e}")
            installed_models = ["No models available"]  # Handle empty list

        return {
            "required": {
                "clip": ("CLIP",),
                "ollama_model": ("COMBO", {"default": installed_models[0], "choices": installed_models}),
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "system_message": ("STRING", {"default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = (
        "conditioning",
        "prompt",
    )
    FUNCTION = "get_encoded"
    CATEGORY = "FluxOllama"
    TITLE = "Ollama CLIP Prompt Encode"

    def get_encoded(self, clip, ollama_url, ollama_model, seed, system_message, text):
        """Gets and encodes the prompt using CLIP."""
        # Generate the prompt from the Ollama API
        prompt = self.generate_prompt(ollama_url, ollama_model, text, system_message, seed)

        # Tokenize and encode the prompt using the provided CLIP model
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        # Return the conditioning and the prompt
        return [[cond, {"pooled_output": pooled}]], prompt
