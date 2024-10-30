from .OllamaPromptGenerator import OllamaPromptGenerator

# Mappings for ComfyUI to recognize the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptGenerator": "Ollama Generator for Vision Capable Models",
}

NODE_CLASS_MAPPINGS = {
    "OllamaPromptGenerator": OllamaPromptGenerator,
}
