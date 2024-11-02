import json
from datetime import datetime
from ollama import Client, Options
import os
from .ollama_helpers import OllamaHelpers


class OllamaPromptGenerator:
    # Defaults
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_SYSTEM_MESSAGE = ("Use the supplied information to create a prompt for a Natural Language Stable "
                             "Diffusion model. Start it with something like \"This is a (art or photo style) of \" "
                             "followed by the supplied elements and additional described elements.  If an image is "
                             "supplied then use the elements in the image to enhance the prompt. Be verbose and "
                             "use full descriptive sentences. Do not describe the prompt. Do not say you are creating "
                             "a prompt. Do not give instructions on creating the prompt.  You are creating the "
                             "prompt, and You are sending it to the model immediately."
                             )
    OLLAMA_SYSTEM_PROMPT = "This is the system prompt, gives the model instructions on how to respond."
    NEGATIVE_PROMPT = "Negative Prompt goes here - Not used if using Flux Models. Text here will not be sent to Ollama."
    POSITIVE_PROMPT = "Positive Prompt - Put your starter prompt and/or tags here.  This will send to Ollama."

    @classmethod
    def INPUT_TYPES(cls):
        model_names = OllamaHelpers.get_available_models()  # Query Ollama API to retrieve installed models
        return {
            "required": {
                "ollama_model": (model_names,),  # Use retrieved model names here.  Will default to first listed.
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "system_message": ("STRING", {"placeholder": cls.OLLAMA_SYSTEM_PROMPT,
                                              "default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "starter_prompt": ("STRING", {"placeholder": cls.POSITIVE_PROMPT, "multiline": True}),
                "neg_prompt": ("STRING", {"placeholder": cls.NEGATIVE_PROMPT, "multiline": True}),
            },
            "optional": {
                "clip": ("CLIP",),
                "input_image": ("IMAGE",),
                "prepend_text": ("STRING", {"forceInput": True}),  # Useful for specific model tags, lora keywords, etc
                "unload_model": ("BOOLEAN", {"default": True}),  # Unloads model to free up VRAM for image generation
                "use_conjoined_prompt": ("BOOLEAN", {"default": False}),  # Append original input to generated prompt
                "log_to_file": ("BOOLEAN", {"default": False}),  # log everything to a textfile
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning+", "conditioning-", "generated_prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "Flux-O-llama"

    @staticmethod
    def process_clip(clip, prompt):
        """Process the prompt with CLIP."""
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]  # Return CLIP conditioning

    def generate_prompt(self, ollama_model, ollama_url, system_message, starter_prompt, neg_prompt, clip=None,
                        input_image=None, prepend_text=None, unload_model=True, use_conjoined_prompt=False,
                        log_to_file=False, seed=None):
        ollama_client = Client(host=ollama_url)
        opts = Options()
        if seed is not None and seed != 0:
            opts["seed"] = seed

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": starter_prompt},
        ]
        messages_string = json.dumps(messages)
        # Handles optional image input for multimodal models, image data ignored by text-only models.
        if input_image is not None:
            # Make API request with the image
            encoded_image = OllamaHelpers.resize_and_encode_image(input_image)
            response = ollama_client.generate(
                model=ollama_model,
                prompt=messages_string,
                options=opts,
                images=[encoded_image],
            )

        else:
            # Call Ollama API to generate the prompt without an image
            response = ollama_client.generate(
                model=ollama_model,
                prompt=messages_string,
                options=opts,
            )

        returned_prompt = response.get("response", "")

        # Extract the prompt from the response
        if prepend_text is not None:
            prepend = prepend_text
        else:
            prepend = ""

        if use_conjoined_prompt:
            starter_prompt = starter_prompt
            generated_response = prepend + ", \n" + returned_prompt + ", \n" + starter_prompt
        else:
            generated_response = prepend + ", \n" + returned_prompt

        if log_to_file:
            try:
                # Create directory if it doesn't exist
                log_dir = os.path.join(os.path.dirname(__file__), 'logs')
                log_file = os.path.join(log_dir, 'logged_prompts.txt')
                os.makedirs(log_dir, exist_ok=True)
                # Get current timestamp
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
                # Open the file in append mode ('a') to add entries without overwriting
                with open(log_file, 'a') as file:
                    # Write variables to the file
                    file.write("Timestamp:      " + str(dt_string) + '\n')
                    file.write("System Prompt:  " + str(system_message) + '\n')
                    file.write("Prepend Text:   " + str(prepend_text) + '\n')
                    file.write("Input Text:     " + str(starter_prompt) + '\n')
                    file.write("Seed:           " + str(seed) + '\n')
                    file.write("Generated Text: " + str(returned_prompt) + '\n\n')
                    print(f"Log entry written to {log_file}")  # Debugging print statement

            except Exception as e:
                print(f"Error while creating log file: {e}")

        conditioning_positive = None
        conditioning_negative = None
        if clip is not None:
            conditioning_positive = self.process_clip(clip, generated_response)
            conditioning_negative = self.process_clip(clip, neg_prompt)
        if unload_model:
            OllamaHelpers.unload_model(ollama_model)
        # Return positive conditioning, negative conditioning, and both prompts for view/save/log nodes.
        return conditioning_positive, conditioning_negative, generated_response
