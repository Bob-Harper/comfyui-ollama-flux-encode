import json

from ollama import Client, Options

from .ollama_helpers import OllamaHelpers


class OllamaPromptGenerator:
    # Defaults
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_SYSTEM_MESSAGE = ("Use the supplied information to create a prompt for a Natural Language Stable "
                             "Diffusion model. Start it with something like \"This is a (art or photo style) of \" "
                             "followed by the supplied elements and additional described elements.  If an image is "
                             "supplied then use the elements in the image to enhance the prompt. Be verbose and "
                             "use full descriptive sentences. Do not describe the prompt. Do not say you are creating "
                             "a prompt. Do not give instructions on creating the prompt.  YOU are creating the "
                             "prompt, and YOU are sending it to the model immediately."
                             )
    NEGATIVE_PROMPT = "Negative Prompt goes here - Not used if using Flux Models. Text here will not be sent to Ollama."
    POSITIVE_PROMPT = "Positive Prompt - Put your starter prompt and/or tags here.  This will send to Ollama."

    @classmethod
    def INPUT_TYPES(cls):
        model_names = OllamaHelpers.get_available_models()  # Query Ollama API to retrieve installed models
        return {
            "required": {
                "ollama_model": (model_names,),  # Use retrieved model names here.  Will default to first listed.
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "system_message": ("STRING", {"default": cls.OLLAMA_SYSTEM_MESSAGE, "multiline": True}),
                "text": ("STRING", {"default": cls.POSITIVE_PROMPT, "multiline": True}),
                "neg_prompt": ("STRING", {"default": cls.NEGATIVE_PROMPT, "multiline": True}),
            },
            "optional": {
                "clip": ("CLIP",),
                "input_image": ("IMAGE",),
                "prepend_text": ("STRING", {"forceInput": True}),  #
                "unload_model": ("BOOLEAN", {"default": True}),  # Unloads model to free up VRAM for image generation
                "use_full_prompt": ("BOOLEAN", {"default": False}),  # Append original input to generated prompt
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning+", "conditioning-", "generated_prompt", "full_prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "Flux-O-llama"

    @staticmethod
    def process_clip(clip, prompt):
        """Process the prompt with CLIP."""
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]  # Return CLIP conditioning

    def generate_prompt(self, ollama_model, ollama_url, system_message, text, neg_prompt, clip=None, input_image=None,
                        prepend_text=None, unload_model=True, use_full_prompt=False, seed=None,):
        ollama_client = Client(host=ollama_url)
        opts = Options()
        if seed is not None and seed != 0:
            opts["seed"] = seed

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text},
        ]
        messages_string = json.dumps(messages)
        # Handles optional image input for multimodal models, image data ignored by text-only models.
        if input_image is not None:
            print("Yes there is an image!")
            # Ensure correct dimensions: 4D tensor [1, channels, height, width]
            try:
                encoded_image = OllamaHelpers.resize_and_encode_image(input_image)
                print("The image is encoded!")
                print(f"This is what the image looks like to the LLM: {encoded_image[:100]}...")

                # Make API request with the image
                response = ollama_client.generate(
                    model=ollama_model,
                    prompt=messages_string,
                    options=opts,
                    images=[encoded_image],  # Pass the encoded image
                )
                print(f"This is the image response from the LLM: {response}")
            except Exception as e:
                print(f"Error processing image: {e}")
                response = {"response": ""}

        else:
            # Call Ollama API to generate the prompt without an image
            response = ollama_client.generate(
                model=ollama_model,
                prompt=messages_string,
                options=opts,
            )
        # Extract the prompt from the response
        if prepend_text is not None:
            prepend = prepend_text
        else:
            prepend = ""
        full_prompt = prepend + response.get("response", "") + f" - initial tags: {text}"
        generated_response = prepend + response.get("response", "")
        conditioning_positive = None
        conditioning_negative = None
        if clip is not None:
            if use_full_prompt:  # Pass the full joined prompt to CLIP
                conditioning_positive = self.process_clip(clip, full_prompt)
                conditioning_negative = self.process_clip(clip, neg_prompt)
            else:  # Pass only the generated prompt to CLIP
                conditioning_positive = self.process_clip(clip, generated_response)
                conditioning_negative = self.process_clip(clip, neg_prompt)
        if unload_model:
            OllamaHelpers.unload_model(ollama_model)
        # Return positive conditioning, negative conditioning, and both prompts for view/save/log nodes.
        return conditioning_positive, conditioning_negative, generated_response, full_prompt


"""
if input_image.ndimension() == 4:
    # Remove the batch dimension and keep only the first 3 channels (RGB)
    input_image = input_image[0, :3, :, :]  # Remove batch and select RGB channels
    # Convert the tensor image to a PIL Image and resize/encode
    print("The image is good!")

else:
    print(f"Unexpected image dimensions: {input_image.ndimension()} dimensions found.")
    response = {"response": ""}    
"""
