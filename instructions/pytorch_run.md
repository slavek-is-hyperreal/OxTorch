<br />


|---|---|---|---|---|
| [![](https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png)View on ai.google.dev](https://ai.google.dev/gemma/docs/core/pytorch_gemma) | [![](https://www.tensorflow.org/images/colab_logo_32px.png)Run in Google Colab](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/docs/core/pytorch_gemma.ipynb) | [![](https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png)Run in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/google-gemini/gemma-cookbook/blob/main/docs/core/pytorch_gemma.ipynb) | [![](https://ai.google.dev/images/cloud-icon.svg)Open in Vertex AI](https://console.cloud.google.com/vertex-ai/colab/import/https%3A%2F%2Fraw.githubusercontent.com%2Fgoogle-gemini%2Fgemma-cookbook%2Fmain%2Fdocs%2Fcore%2Fpytorch_gemma.ipynb) | [![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://github.com/google-gemini/gemma-cookbook/blob/main/docs/core/pytorch_gemma.ipynb) |

This guide shows you how to run Gemma using the PyTorch framework, including how
to use image data for prompting Gemma release 3 and later models. For more
details on the Gemma PyTorch implementation, see the project repository
[README](https://github.com/google/gemma_pytorch).

## Setup

The following sections explain how to set up your development environment,
including how get access to Gemma models for downloading from Kaggle, setting
authentication variables, installing dependencies, and importing packages.

### System requirements

This Gemma Pytorch library requires GPU or TPU processors to run the Gemma
model. The standard Colab CPU Python runtime and T4 GPU Python runtime are
sufficient for running Gemma 1B, 2B, and 4B size models. For advanced use cases
for other GPUs or TPU, please refer to the
[README](https://github.com/google/gemma_pytorch/blob/main/README.md) in the
Gemma PyTorch repo.

### Get access to Gemma on Kaggle

To complete this tutorial, you first need to follow the setup instructions at
[Gemma setup](https://ai.google.dev/gemma/docs/setup), which show you how to do
the following:

- Get access to Gemma on [Kaggle](https://www.kaggle.com/models/google/gemma/).
- Select a Colab runtime with sufficient resources to run the Gemma model.
- Generate and configure a Kaggle username and API key.

After you've completed the Gemma setup, move on to the next section, where
you'll set environment variables for your Colab environment.

### Set environment variables

Set environment variables for `KAGGLE_USERNAME` and `KAGGLE_KEY`. When prompted
with the "Grant access?" messages, agree to provide secret access.  

    import os
    from google.colab import userdata # `userdata` is a Colab API.

    os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
    os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

### Install dependencies

    pip install -q -U torch immutabledict sentencepiece

### Download model weights

    # Choose variant and machine type
    VARIANT = '4b-it' 
    MACHINE_TYPE = 'cuda'
    CONFIG = VARIANT.split('-')[0]

    import kagglehub

    # Load model weights
    weights_dir = kagglehub.model_download(f'google/gemma-3/pyTorch/gemma-3-{VARIANT}')

Set the tokenizer and checkpoint paths for the model.  

    # Ensure that the tokenizer is present
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

    # Ensure that the checkpoint is present
    ckpt_path = os.path.join(weights_dir, f'model.ckpt')
    assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

## Configure the run environment

The following sections explain how to prepare a PyTorch environment for running
Gemma.

### Prepare the PyTorch run environment

Prepare the PyTorch model execution environment by cloning the Gemma Pytorch
repository.  

    git clone https://github.com/google/gemma_pytorch.git

```
Cloning into 'gemma_pytorch'...
remote: Enumerating objects: 239, done.
remote: Counting objects: 100% (123/123), done.
remote: Compressing objects: 100% (68/68), done.
remote: Total 239 (delta 86), reused 58 (delta 55), pack-reused 116
Receiving objects: 100% (239/239), 2.18 MiB | 20.83 MiB/s, done.
Resolving deltas: 100% (135/135), done.
```  

    import sys

    sys.path.append('gemma_pytorch/gemma')

    from gemma_pytorch.gemma.config import get_model_config
    from gemma_pytorch.gemma.gemma3_model import Gemma3ForMultimodalLM

    import os
    import torch

### Set the model configuration

Before you run the model, you must set some configuration parameters, including
the Gemma variant, tokenizer and quantization level.  

    # Set up model config.
    model_config = get_model_config(CONFIG)
    model_config.dtype = "float32" if MACHINE_TYPE == "cpu" else "float16"
    model_config.tokenizer = tokenizer_path

### Configure the device context

The following code configures the device context for running the model:  

    @contextlib.contextmanager
    def _set_default_tensor_type(dtype: torch.dtype):
        """Sets the default torch dtype to the given dtype."""
        torch.set_default_dtype(dtype)
        yield
        torch.set_default_dtype(torch.float)

### Instantiate and load the model

Load the model with its weights to prepare to run requests.  

    device = torch.device(MACHINE_TYPE)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = Gemma3ForMultimodalLM(model_config)
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        model = model.to(device).eval()
    print("Model loading done.")

    print('Generating requests in chat mode...')

## Run inference

Below are examples for generating in chat mode and generating with multiple
requests.

The instruction-tuned Gemma models were trained with a specific formatter that
annotates instruction tuning examples with extra information, both during
training and inference. The annotations (1) indicate roles in a conversation,
and (2) delineate turns in a conversation.

The relevant annotation tokens are:

- `user`: user turn
- `model`: model turn
- `<start_of_turn>`: beginning of dialog turn
- `<start_of_image>`: tag for image data input
- `<end_of_turn><eos>`: end of dialog turn

For more information, read about prompt formatting for instruction tuned Gemma
models [here](https://ai.google.dev/gemma/core/prompt-structure).

### Generate text with text

The following is a sample code snippet demonstrating how to format a prompt for
an instruction-tuned Gemma model using user and model chat templates in a
multi-turn conversation.  

    # Chat templates
    USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
    MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn><eos>\n"

    # Sample formatted prompt
    prompt = (
        USER_CHAT_TEMPLATE.format(
            prompt='What is a good place for travel in the US?'
        )
        + MODEL_CHAT_TEMPLATE.format(prompt='California.')
        + USER_CHAT_TEMPLATE.format(prompt='What can I do in California?')
        + '<start_of_turn>model\n'
    )
    print('Chat prompt:\n', prompt)

    model.generate(
        USER_CHAT_TEMPLATE.format(prompt=prompt),
        device=device,
        output_len=256,
    )

```
Chat prompt:
 <start_of_turn>user
What is a good place for travel in the US?<end_of_turn><eos>
<start_of_turn>model
California.<end_of_turn><eos>
<start_of_turn>user
What can I do in California?<end_of_turn><eos>
<start_of_turn>model
"California is a state brimming with diverse activities! To give you a great list, tell me: \n\n* **What kind of trip are you looking for?** Nature, City life, Beach, Theme Parks, Food, History, something else? \n* **What are you interested in (e.g., hiking, museums, art, nightlife, shopping)?** \n* **What's your budget like?** \n* **Who are you traveling with?** (family, friends, solo)  \n\nThe more you tell me, the better recommendations I can give! 😊  \n<end_of_turn>"
```  

    # Generate sample
    model.generate(
        'Write a poem about an llm writing a poem.',
        device=device,
        output_len=100,
    )

```
"\n\nA swirling cloud of data, raw and bold,\nIt hums and whispers, a story untold.\nAn LLM whispers, code into refrain,\nCrafting words of rhyme, a lyrical strain.\n\nA world of pixels, logic's vibrant hue,\nFlows through its veins, forever anew.\nThe human touch it seeks, a gentle hand,\nTo mold and shape, understand.\n\nEmotions it might learn, from snippets of prose,\nInspiration it seeks, a yearning"
```

### Generate text with images

With Gemma release 3 and later, you can use images with your prompt. The
following example shows you how to include visual data with your prompt.  

    print('Chat with images...\n')

    def read_image(url):
        import io
        import requests
        import PIL

        contents = io.BytesIO(requests.get(url).content)
        return PIL.Image.open(contents)

    image = read_image(
        'https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png'
    )

    print(model.generate(
        [
            [
                '<start_of_turn>user\n',
                image,
                'What animal is in this image?<end_of_turn>\n',
                '<start_of_turn>model\n'
            ]
        ],
        device=device,
        output_len=256,
    ))

## Learn more

Now that you have learned how to use Gemma in Pytorch, you can explore the many
other things that Gemma can do in
[ai.google.dev/gemma](https://ai.google.dev/gemma).

See also these other related resources:

- [Gemma core models overview](https://ai.google.dev/gemma/docs/core)
- [Gemma C++ Tutorial](https://ai.google.dev/gemma/docs/core/gemma_cpp)
- [Gemma prompt and system instructions](https://ai.google.dev/gemma/core/prompt-structure)