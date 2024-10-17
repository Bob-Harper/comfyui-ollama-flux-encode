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
        installed_models = cls().list_installed_models(cls.OLLAMA_URL)  # Get available models
        return {
            "required": {
                "clip": ("CLIP",),
                "ollama_model": ("STRING", {"default": installed_models[0], "choices": installed_models}),  # Dropdown with installed models
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

    CATEGORY = "Ollama"

    def sanitize_prompt(self, prompt):
        """Sanitize the prompt for use in clip encoding."""
        return prompt.replace(".", ",")

    def get_encoded(self, clip, ollama_url, ollama_model, seed, system_message, text):
        """Gets and encodes the prompt using CLIP."""
        # Fetch and sanitize the prompt using inherited method
        combined_prompt = self.get_prompt(ollama_url, ollama_model, seed, system_message, text)[0]

        # Tokenize and encode the prompt with CLIP
        tokens = clip.tokenize(combined_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return ([[cond, {"pooled_output": pooled}]], combined_prompt)
