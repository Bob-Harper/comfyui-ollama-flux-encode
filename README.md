# ComfyUI Generator for Vision Capable Models

** forked to provide a personalized version that will work with Flux Models

The original version of these nodes was set up for tags and short descriptive words.  Flux excels at natural language interpretation.

Being that i almost exclusively use Flux - here we are.

A prompt generator and CLIP encoder using AI provided by [Ollama](https://ollama.com).

## Prerequisites

Install [Ollama](https://ollama.com) and have the service running.

This node has been tested with ollama version `0.3.12`.

## Installation

Choose one of the following methods to install the node:

### via ComfyUI Manager

If you have the [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, you can install the node from the `install via git url`.

### via Git

Clone this repository into your `<comfyui>/custom_nodes` directory.

```sh
git clone https://github.com/Bob-Harper/comfyui-ollama-flux-encode.git
```

## Usage

The `Ollama CLIP Prompt Encode` node is designed to replace the default `CLIP Text Encode (Prompt)` node. It generates a prompt using the Ollama AI model and then encodes the prompt with CLIP.

The node will output the generated prompt as a `string`. This can be viewed with a node that will display text.  (I prefer pyssssss "Show Text")  I have considered having it show in the node itself, and I still may come back to that.  Or have an option to have the generated prompt print to console.  Again, i may come back to that.

An example workflow *may* be available in the `docs` folder.  i'll update that soon enough.

### Ollama URL

The URL to the Ollama service. The default is `http://localhost:11434`.

### Ollama Model

This is the model that is used to generate your prompt.

Some models that work well with this prompt generator are:

- `orca-mini`
- `tinyllama`

### NEW FEATURE: MULTIMODAL SUPPORT

If a multimodal LLM is selected, and an optional image is supplied to the input node, it will send both the text input and the image to the model for analasys.  This should make for some interesting Img2Img results.  

Suggested models:

- `llava-llama3`
- `mannix/llava-phi3:iq2_s `

(The former is brilliant at analysis, the latter is far less prudish and doesn't freak out about illegal content just because someone is wearing a towel)

I switched it up so the node will now show you the currently installed models in dropdown.  

Want a new model?  Install with ollama pull in a terminal. Restart the server and it will refresh the node dropdown. 

Smaller models are recommended for faster generation times (tinyllama is great for this), though llava models are not particularly small.  Tradeoff is that it CAN see the image.  You decide for yourself if it's worth the tradeoff.

### Seed

The seed that will be used to generate the prompt. This is useful for generating the same prompt multiple times or ensuring a different prompt is generated each time.

### Prompting

It does have a hardcoded system prompt as default, targeted for FLux style prompting.  

Any edits to the system prompt in the text box will override the default.

This should allow for future changes to prompting methods without needing to recode, and can still be reworded for non-Flux models.

If there are tags that you want to send through as is - there is an option to send Full Prompt.  If anyone actually uses this and asks, i could add a Prepend/Append option for the fully joined prompt.

Right now it will default to sending the Generated Prompt Only through to the CLIP encode.  Switching to Use Full Prompt will send your original text appended to the Generated text.

Note that the negative prompt is not used with flux, entering text in that field will nave no effect.  If using this with a non Flux model, have at it, negative-prompt your heart out.  It will be sent to CLIP and then to the model, but will not be included in the text sent to Ollama for prompt generation.

## Credits

(credit where it is due, of course.  I may have my preferences and ideas on improvements, but this wouldn't exist without a solid implementation to work from.)

[Michael Standen](https://michael.standen.link)

This software is provided under the [MIT License](https://tldrlegal.com/license/mit-license) so it's free to use so long as you give me (Bob) and [Michael Standen](https://michael.standen.link) credit.
