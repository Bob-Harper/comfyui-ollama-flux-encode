from .OllamaPromptGenerator import OllamaPromptGenerator
from .OllamaModelSelector import OllamaModelSelector  # Import the new model selector

# Mappings for ComfyUI to recognize the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptGenerator": "Ollama Prompt Generator",
    "OllamaModelSelector": "Ollama Model Selector",  # Add new model selector
}

NODE_CLASS_MAPPINGS = {
    "OllamaPromptGenerator": OllamaPromptGenerator,
    "OllamaModelSelector": OllamaModelSelector,  # Map the new model selector class
}
