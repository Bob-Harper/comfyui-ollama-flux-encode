from .ollama_helpers import available_models


class OllamaModelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}  # No inputs needed for this node

    @classmethod
    def OUTPUT_TYPES(cls):
        return {"OllamaModelName": ("STRING",)}  # Output the selected model name as a string

    @classmethod
    def get_available_models(cls):
        models_list = available_models()  # Assuming this is a function that returns the list of models
        if not models_list:
            return ["No models available"]
        return models_list

    @classmethod
    def generate_output(cls):
        models = cls.get_available_models()
        return {"OllamaModelName": models}

    # Add RETURN_TYPES and RETURN_NAMES attributes
    RETURN_TYPES = ("STRING",)  # Specify the return type
    RETURN_NAMES = ("OllamaModelName",)  # Specify the name for the return type

    CATEGORY = "Flux-O-llama"
    TITLE = "Ollama Model Selector"

