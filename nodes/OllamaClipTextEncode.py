"""
@author: Bob Harper (building on the work of Michael Standen)
@title: Ollama Flux Prompt Encode
@nickname: Ollama Flux Prompt Encode
@description: Use AI to generate Flux style prompts and perform CLIP text encoding
"""
from .OllamaPromptGenerator import OllamaPromptGenerator


class OllamaCLIPTextEncode(OllamaPromptGenerator):
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

    @classmethod
    def INPUT_TYPES(cls):
        model_list = cls.list_installed_models(cls.OLLAMA_URL)  # Consistent method for fetching models
        if not model_list:
            model_list = ["No models available"]  # Fallback option

        return {
            "required": {
                "clip": ("CLIP",),
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "ollama_model": ("COMBO", {"default": model_list[0], "choices": model_list}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "system_message": ("STRING", {"default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    def get_encoded(self, clip, ollama_url, ollama_model, seed, prepend_tags, text):
        """Gets and encodes the prompt using CLIP."""
        combined_prompt = self.generate_prompt(ollama_url, ollama_model, seed, prepend_tags, text)[0]
        tokens = clip.tokenize(combined_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]], combined_prompt
