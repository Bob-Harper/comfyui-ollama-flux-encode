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
        return {
            "required": {
                "clip": ("CLIP",),
                "ollama_model": ("STRING", {"default": cls.OLLAMA_MODEL}),  # Hardcoded model field
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "system_message": ("STRING", {"default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "populated_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "get_encoded"
    CATEGORY = "FluxOllama"
    TITLE = "Ollama Flux CLIP Prompt Encoder"

    def get_encoded(self, clip, ollama_url, ollama_model, seed, system_message, text):
        """Gets and encodes the prompt using CLIP."""
        # Generate the prompt from the Ollama API
        prompt = self.generate_prompt(ollama_url, ollama_model, text, system_message, seed)

        # Tokenize and encode the prompt using the provided CLIP model
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        # Return the conditioning and the prompt
        return [[cond, {"pooled_output": pooled}]], prompt
