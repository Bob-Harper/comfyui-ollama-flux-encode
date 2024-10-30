"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use AI to generate prompts and perform CLIP text encoding
"""

from OllamaPromptGenerator import OllamaPromptGenerator

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptGenerator": "Ollama Prompt Generator",
}

NODE_CLASS_MAPPINGS = {
    "OllamaPromptGenerator": OllamaPromptGenerator,
}
