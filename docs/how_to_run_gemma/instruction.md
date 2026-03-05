# Gemma models overview

![Alt Gemma logo](https://ai.google.dev/images/gemma_sq.png)
Gemma is a family of lightweight, state-of-the-art open models built from the
same research and technology used to create the
[Gemini](https://deepmind.google/technologies/gemini/#introduction) models.
Developed by Google DeepMind and other teams across Google, Gemma is named after
the Latin *gemma*, meaning precious stone. The Gemma model weights are
supported by developer tools that promote innovation, collaboration, and the
responsible use of artificial intelligence (AI). You can get multiple variations
of Gemma for general and specific use cases:

- [**Gemma 3**](https://ai.google.dev/gemma/docs/core): Solve a wide variety of generative AI tasks with text and image input, support for over 140 languages, and long 128K context window.
- [**CodeGemma**](https://ai.google.dev/gemma/docs/codegemma): Complete programming tasks with this lightweight, coding-focused generative model.
- [**PaliGemma 2**](https://ai.google.dev/gemma/docs/paligemma): Build visual data processing AI solutions with a model that's built to be fine-tuned for your image data processing applications and available in multiple resolutions.
- [**ShieldGemma 2**](https://ai.google.dev/gemma/docs/shieldgemma): Evaluate the safety of generative AI models' input and output against defined policies.

Many more Gemma variants are available from Google and our AI developer
community. Check them out on
[Kaggle Models](https://www.kaggle.com/models?query=gemma) and
[Hugging Face](https://huggingface.co/models?search=google/gemma).
Get inspired by what our community members have built with Gemma in the
[Gemmaverse](https://ai.google.dev/gemma/gemmaverse).

The Gemma models are available to run in
your applications and on your hardware, mobile devices, or hosted services. You
can also customize these models using tuning techniques so that they excel at
performing specific tasks that matter to you and your users. Gemma models draw
inspiration and technological lineage from the Gemini family of models, and are
made for the AI development community to extend and take further.

Ready to begin? **[Get started](https://ai.google.dev/gemma/docs/get_started)** with Gemma models!


[Video](https://www.youtube.com/watch?v=qcjrduz_YS8)

The Gemma family of open models includes a range of model sizes, capabilities,
and task-specialized variations to help you build custom generative solutions.
These are the main paths you can follow when using Gemma models in an
application:

- Select a model and **deploy it as-is** in your application
- Select a model, **tune it for a specific task**, and then deploy it in an application, or share it with the community.

This guide helps you get started with [picking](https://ai.google.dev/gemma/docs/get_started#pick) a model, [testing](https://ai.google.dev/gemma/docs/get_started#test)
its capabilities, and optionally, [tuning](https://ai.google.dev/gemma/docs/get_started#tune) the model you selected for
your application.
| **Tip:** As you begin implementing AI applications, make sure you are following a principled approach to AI that serves all your users with the [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).

[Get it on Kaggle](https://www.kaggle.com/models?query=gemma3&publisher=google)
[Get it on Hugging Face](https://huggingface.co/models?search=google/gemma-3)

## Pick a model

This section helps you understand the official variants of the Gemma model
family and select a model for your application. The model variants provide
general capabilities or are specialized for specific tasks, and are provided
in different parameter sizes so you can pick a model that has your preferred
capabilities and meets your compute requirements.
| **Tip:** A good place to start is the [Gemma 3 4B](https://www.kaggle.com/models/google/gemma-3) model in the latest available version, which can be used for many tasks and has lower resource requirements.

### Gemma models list

The following table lists the major variants of the Gemma model family and their
intended deployment platforms:

| **Parameter size** | **Input** | **Output** | **Variant** | **Foundation** | **Intended platforms** |
|---|---|---|---|---|---|
| 270M | Text | Text | - [Gemma 3 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3) | Mobile devices and single board computers |
| 1B | Text | Text | - [Gemma 3 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3) | Mobile devices and single board computers |
| E2B | Text, images, audio | Text | - [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n) | [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n/model_card) | Mobile devices |
| 2B | Text | Text | - [Gemma 2 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) | Mobile devices and laptops |
| 2B | Text | Text | - [Gemma (core)](https://ai.google.dev/gemma/docs/core) - [CodeGemma](https://ai.google.dev/gemma/docs/codegemma) | [Gemma 1](https://ai.google.dev/gemma/docs/core/model_card) | Mobile devices and laptops |
| 3B | Text, images | Text | - [PaliGemma 2](https://ai.google.dev/gemma/docs/paligemma) | [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) | Desktop computers and small servers |
| E4B | Text, images, audio | Text | - [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n) | [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n/model_card) | Mobile devices and laptops |
| 4B | Text, images | Text | - [Gemma 3 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3) | Desktop computers and small servers |
| 7B | Text | Text | - [Gemma (core)](https://ai.google.dev/gemma/docs/core) - [CodeGemma](https://ai.google.dev/gemma/docs/codegemma) | [Gemma 1](https://ai.google.dev/gemma/docs/core/model_card) | Desktop computers and small servers |
| 9B | Text | Text | - [Gemma 2 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) | Higher-end desktop computers and servers |
| 10B | Text, images | Text | - [PaliGemma 2](https://ai.google.dev/gemma/docs/paligemma) | [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) | Higher-end desktop computers and servers |
| 12B | Text, images | Text | - [Gemma 3 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3) | Higher-end desktop computers and servers |
| 27B | Text, images | Text | - [Gemma 3 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3) | Large servers or server clusters |
| 27B | Text | Text | - [Gemma 2 (core)](https://ai.google.dev/gemma/docs/core) | [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) | Large servers or server clusters |
| 28B | Text, images | Text | - [PaliGemma 2](https://ai.google.dev/gemma/docs/paligemma) | [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) | Large servers or server clusters |

The Gemma family of models also includes special-purpose and research models,
including
[ShieldGemma](https://ai.google.dev/gemma/docs/shieldgemma),
[DataGemma](https://ai.google.dev/gemma/docs/datagemma),
[Gemma Scope](https://ai.google.dev/gemma/docs/gemmascope),
and
[Gemma-APS](https://ai.google.dev/gemma/docs/gemma-aps).
| **Tip:** You can download official Google Gemma model variants and community-created variants from [Kaggle Models](https://www.kaggle.com/models?query=gemma) and [Hugging Face](https://huggingface.co/models?search=google/gemma).

## Test models

You can test Gemma models by setting up a development environment with a
downloaded model and supporting software. You can then prompt the model and
evaluate its responses. Use one of the following Python notebooks with your
preferred machine learning framework to set up a testing environment and prompt
a Gemma model:

- [Inference with Keras](https://ai.google.dev/gemma/docs/core/keras_inference)
- [Inference with PyTorch](https://ai.google.dev/gemma/docs/core/pytorch_gemma)
- [Inference with Gemma library](https://ai.google.dev/gemma/docs/core/gemma_library)

## Tune models

You can change the behavior of Gemma models by performing tuning on them. Tuning
a model requires a dataset of inputs and expected responses of sufficient size
and variation to guide the behavior of the model. You also need significantly
more computing and memory resources to complete a tuning run compared to running
a Gemma model for text generation. Use one of the following Python notebooks to
set up a tuning development environment and tune a Gemma model:

- [Tune Gemma with Keras and LoRA tuning](https://ai.google.dev/gemma/docs/core/lora_tuning)
- [Tune larger Gemma models with distributed training](https://ai.google.dev/gemma/docs/core/distributed_tuning)

## Next Steps

Check out these guides for building more solutions with Gemma:

- [Create a chatbot with Gemma](https://ai.google.dev/gemma/docs/gemma_chat)
- [Deploy Gemma to production with Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma)
- [Use Genkit with Ollama and Gemma](https://genkit.dev/docs/integrations/ollama/)

This page documents releases for the Gemma family of models.

## January 15, 2026

- Release of [TranslateGemma](https://www.kaggle.com/models/google/translategemma) in 4B, 12B, and 27B parameter size.

## January 13, 2026

- Release of [MedGemma 1.5](https://developers.google.com/health-ai-developer-foundations/medgemma) in 4B parameter size.

## December 19, 2025

- Release of [Gemma Scope 2](https://ai.google.dev/gemma/docs/gemma_scope), an interpretability suite for Gemma 3 models.

## December 18, 2025

- Release of [FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma) in 270M parameter size.
- Release of [T5Gemma v2](https://blog.google/technology/developers/t5gemma-2/) in 270M-270M, 1B-1B, and 4B-4B parameter sizes.

## September 13, 2025

- Release of [VaultGemma](https://www.kaggle.com/models/google/vaultgemma) in 1B parameter size.

## September 4, 2025

- Release of [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma) in 308M parameter size.

## August 14, 2025

- Release of [Gemma 3](https://ai.google.dev/gemma/docs/core) in 270M size.

## July 9, 2025

- Release of [T5Gemma](https://developers.googleblog.com/en/t5gemma/) across different parameter sizes.
- Release of [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma) 27B parameter multimodal model.

## June 26, 2025

- Release of [Gemma 3n](https://ai.google.dev/gemma/docs/3n) in E2B and E4B sizes.

## May 20, 2025

- Release of [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma) in 4B and 27B parameter sizes.

## March 10, 2025

- Release of [Gemma 3](https://ai.google.dev/gemma/docs/core) in 1B, 4B, 12B and 27B sizes.
- Release of [ShieldGemma 2](https://ai.google.dev/gemma/docs/shieldgemma).

## February 19, 2025

- Release of [PaliGemma 2 mix](https://ai.google.dev/gemma/docs/paligemma/model-card-2) in 3B, 10B, and 28B parameter sizes.

## December 5, 2024

- Release of [PaliGemma 2](https://ai.google.dev/gemma/docs/paligemma) in 3B, 10B, and 28B parameter sizes.

## October 16, 2024

- Release of [Personal AI code assistant](https://ai.google.dev/gemma/docs/personal-code-assistant) developer guide.

## October 15, 2024

- Release of [Gemma-APS](https://ai.google.dev/gemma/docs/gemma-aps) in 2B and 7B sizes.

## October 8, 2024

- Release of [Business email assistant](https://ai.google.dev/gemma/docs/business-email-assistant) developer guide.

## October 3, 2024

- Release of [Gemma 2 JPN](https://www.kaggle.com/models/google/gemma-2-2b-jpn-it) in 2B size.
- Release of [Spoken language tasks](https://ai.google.dev/gemma/docs/spoken-language/task-specific-tuning) developer guide.

## September 12, 2024

- Release of [DataGemma](https://ai.google.dev/gemma/docs/datagemma) in 2B size.

## July 31, 2024

- Release of [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) in 2B size.
- Initial release of [ShieldGemma](https://ai.google.dev/gemma/docs/shieldgemma).
- Initial release of [Gemma Scope](https://ai.google.dev/gemma/docs/gemma_scope).

## June 27, 2024

- Initial release of [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) in 9B and 27B sizes.

## June 11, 2024

- Release of [RecurrentGemma](https://ai.google.dev/gemma/docs/recurrentgemma) 9B variant.

## May 14, 2024

- Initial release of [PaliGemma](https://ai.google.dev/gemma/docs/paligemma).

## May 3, 2024

- Release of [CodeGemma](https://ai.google.dev/gemma/docs/codegemma) v1.1.

## April 9, 2024

- Initial release of [CodeGemma](https://ai.google.dev/gemma/docs/codegemma).
- Initial release of [RecurrentGemma](https://ai.google.dev/gemma/docs/recurrentgemma).

## April 5, 2024

- Release of [Gemma](https://ai.google.dev/gemma/docs) 1.1.

## February 21, 2024

- Initial release of [Gemma](https://ai.google.dev/gemma/docs) in 2B and 7B sizes.

<br />

Gemma 3n is a generative AI model optimized for use in everyday devices, such as phones, laptops, and tablets. This model includes innovations in parameter-efficient processing, including Per-Layer Embedding (PLE) parameter caching and a MatFormer model architecture that provides the flexibility to reduce compute and memory requirements. These models feature audio input handling, as well as text and visual data.

Gemma 3n includes the following key features:

- **Audio input** : Process sound data for speech recognition, translation, and audio data analysis.[Learn more](https://ai.google.dev/gemma/docs/core/huggingface_inference#audio)
- **Visual and text input** : Multimodal capabilities let you handle vision, sound, and text to help you understand and analyze the world around you.[Learn more](https://ai.google.dev/gemma/docs/core/huggingface_inference#vision)
- **Vision encoder:** High-performance MobileNet-V5 encoder substantially improves speed and accuracy of processing visual data.[Learn more](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/#mobilenet-v5:-new-state-of-the-art-vision-encoder)
- **PLE caching** : Per-Layer Embedding (PLE) parameters contained in these models can be cached to fast, local storage to reduce model memory run costs.[Learn more](https://ai.google.dev/gemma/docs/gemma-3n#ple-caching)
- **MatFormer architecture:** Matryoshka Transformer architecture allows for selective activation of the models parameters per request to reduce compute cost and response times.[Learn more](https://ai.google.dev/gemma/docs/gemma-3n#matformer)
- **Conditional parameter loading:** Bypass loading of vision and audio parameters in the model to reduce the total number of loaded parameters and save memory resources.[Learn more](https://ai.google.dev/gemma/docs/gemma-3n#conditional-parameter)
- **Wide language support**: Wide linguistic capabilities, trained in over 140 languages.
- **32K token context**: Substantial input context for analyzing data and handling processing tasks.

[Get it on Kaggle](https://www.kaggle.com/models/google/gemma-3n)[Get it on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)

As with other Gemma models, Gemma 3n is provided with open weights and licensed for responsible[commercial use](https://ai.google.dev/gemma/terms), allowing you to tune and deploy it in your own projects and applications.
| **Tip:** If you are interested in building generative AI solutions for Android mobile applications, check out Gemini Nano. For more information, see the Android[Gemini Nano](https://developer.android.com/ai/gemini-nano)developer docs.

## Model parameters and effective parameters

Gemma 3n models are listed with parameter counts, such as**`E2B`** and**`E4B`** , that are*lower* than the total number of parameters contained in the models. The**`E`**prefix indicates these models can operate with a reduced set of Effective parameters. This reduced parameter operation can be achieved using the flexible parameter technology built into Gemma 3n models to help them run efficiently on lower resource devices.

The parameters in Gemma 3n models are divided into 4 main groups: text, visual, audio, and per-layer embedding (PLE) parameters. With standard execution of the E2B model, over 5 billion parameters are loaded when executing the model. However, using parameter skipping and PLE caching techniques, this model can be operated with an effective memory load of just under 2 billion (1.91B) parameters, as illustrated in Figure 1.

![Gemma 3n diagram of parameter usage](https://ai.google.dev/static/gemma/docs/images/gemma-3n-parameters.png)

**Figure 1.**Gemma 3n E2B model parameters running in standard execution versus an effectively lower parameter load using PLE caching and parameter skipping techniques.

Using these parameter offloading and selective activation techniques, you can run the model with a very lean set of parameters or activate additional parameters to handle other data types such as visual and audio. These features enable you to ramp up model functionality or ramp down capabilities based on device capabilities or task requirements. The following sections explain more about the parameter efficient techniques available in Gemma 3n models.

## PLE caching

Gemma 3n models include Per-Layer Embedding (PLE) parameters that are used during model execution to create data that enhances the performance of each model layer. The PLE data can be generated separately, outside the operating memory of the model, cached to fast storage, and then added to the model inference process as each layer runs. This approach allows PLE parameters to be kept out of the model memory space, reducing resource consumption while still improving model response quality.

## MatFormer architecture

Gemma 3n models use a Matryoshka Transformer or*MatFormer* model architecture that contains nested, smaller models within a single, larger model. The nested sub-models can be used for inferences without activating the parameters of the enclosing models when responding to requests. This ability to run just the smaller, core models within a MatFormer model can reduce compute cost, and response time, and energy footprint for the model. In the case of Gemma 3n, the E4B model contains the parameters of the E2B model. This architecture also lets you select parameters and assemble models in intermediate sizes between 2B and 4B. For more details on this approach, see the[MatFormer research paper](https://arxiv.org/pdf/2310.07707). Try using MatFormer techniques to reduce the size of a Gemma 3n model with the[MatFormer Lab](https://goo.gle/gemma3n-matformer-lab)guide.

## Conditional parameter loading

Similar to PLE parameters, you can skip loading of some parameters into memory, such as audio or visual parameters, in the Gemma 3n model to reduce memory load. These parameters can be dynamically loaded at runtime if the device has the required resources. Overall, parameter skipping can further reduce the required operating memory for a Gemma 3n model, enabling execution on a wider range of devices and allowing developers to increase resource efficiency for less demanding tasks.

<br />

Ready to start building?[Get started](https://ai.google.dev/gemma/docs/get_started)with Gemma models!


**Model Page** : [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n)

**Resources and Technical Documentation**:

- [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
- [Gemma on Kaggle](https://www.kaggle.com/models/google/gemma-3n)
- [Gemma on HuggingFace](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)
- [Gemma on Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemma3n)

**Terms of Use** : [Terms](https://ai.google.dev/gemma/terms)  

**Authors**: Google DeepMind

## Model Information

Summary description and brief definition of inputs and outputs.

### Description

Gemma is a family of lightweight, state-of-the-art open models from Google,
built from the same research and technology used to create the Gemini models.
Gemma 3n models are designed for efficient execution on low-resource devices.
They are capable of multimodal input, handling text, image, video, and audio
input, and generating text outputs, with open weights for pre-trained and
instruction-tuned variants. These models were trained with data in over 140
spoken languages.

Gemma 3n models use selective parameter activation technology to reduce resource
requirements. This technique allows the models to operate at an effective size
of 2B and 4B parameters, which is lower than the total number of parameters they
contain. For more information on Gemma 3n's efficient parameter management
technology, see the
[Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n#parameters)
page.

### Inputs and outputs

- **Input:**
  - Text string, such as a question, a prompt, or a document to be summarized
  - Images, normalized to 256x256, 512x512, or 768x768 resolution and encoded to 256 tokens each
  - Audio data encoded to 6.25 tokens per second from a single channel
  - Total input context of 32K tokens
- **Output:**
  - Generated text in response to the input, such as an answer to a question, analysis of image content, or a summary of a document
  - Total output length up to 32K tokens, subtracting the request input tokens

### Citation

    @article{gemma_3n_2025,
        title={Gemma 3n},
        url={https://ai.google.dev/gemma/docs/gemma-3n},
        publisher={Google DeepMind},
        author={Gemma Team},
        year={2025}
    }

## Model Data

Data used for model training and how the data was processed.

### Training Dataset

These models were trained on a dataset that includes a wide variety of sources
totalling approximately 11 trillion tokens. The knowledge cutoff date for the
training data was June 2024. Here are the key components:

- **Web Documents**: A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. The training dataset includes content in over 140 languages.
- **Code**: Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to generate code and understand code-related questions.
- **Mathematics**: Training on mathematical text helps the model learn logical reasoning, symbolic representation, and to address mathematical queries.
- **Images**: A wide range of images enables the model to perform image analysis and visual data extraction tasks.
- Audio: A diverse set of sound samples enables the model to recognize speech, transcribe text from recordings, and identify information in audio data.

The combination of these diverse data sources is crucial for training a
powerful multimodal model that can handle a wide variety of different tasks and
data formats.

### Data Preprocessing

Here are the key data cleaning and filtering methods applied to the training
data:

- **CSAM Filtering**: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.
- **Sensitive Data Filtering**: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.
- **Additional methods** : Filtering based on content quality and safety in line with [our policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

## Implementation Information

Details about the model internals.

### Hardware

Gemma was trained using [Tensor Processing Unit
(TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) hardware (TPUv4p, TPUv5p
and TPUv5e). Training generative models requires significant computational
power. TPUs, designed specifically for matrix operations common in machine
learning, offer several advantages in this domain:

- **Performance**: TPUs are specifically designed to handle the massive computations involved in training generative models. They can speed up training considerably compared to CPUs.
- **Memory**: TPUs often come with large amounts of high-bandwidth memory, allowing for the handling of large models and batch sizes during training. This can lead to better model quality.
- **Scalability**: TPU Pods (large clusters of TPUs) provide a scalable solution for handling the growing complexity of large foundation models. You can distribute training across multiple TPU devices for faster and more efficient processing.
- **Cost-effectiveness**: In many scenarios, TPUs can provide a more cost-effective solution for training large models compared to CPU-based infrastructure, especially when considering the time and resources saved due to faster training.

These advantages are aligned with
[Google's commitments to operate sustainably](https://sustainability.google/operating-sustainably/).

### Software

Training was done using [JAX](https://github.com/jax-ml/jax) and
[ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/).
JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models. ML
Pathways is Google's latest effort to build artificially intelligent systems
capable of generalizing across multiple tasks. This is specially suitable for
foundation models, including large language models like these ones.

Together, JAX and ML Pathways are used as described in the
[paper about the Gemini family of models](https://goo.gle/gemma2report):
*"the 'single controller' programming model of Jax and Pathways allows a single
Python process to orchestrate the entire training run, dramatically simplifying
the development workflow."*

## Evaluation

Model evaluation metrics and results.

### Benchmark Results

These models were evaluated at full precision (float32) against a large
collection of different datasets and metrics to cover different aspects of
content generation. Evaluation results marked with **IT** are for
instruction-tuned models. Evaluation results marked with **PT** are for
pre-trained models.

#### Reasoning and factuality

| Benchmark | Metric | n-shot | E2B PT | E4B PT |
|---|---|---|---|---|
| [HellaSwag](https://arxiv.org/abs/1905.07830) | Accuracy | 10-shot | 72.2 | 78.6 |
| [BoolQ](https://arxiv.org/abs/1905.10044) | Accuracy | 0-shot | 76.4 | 81.6 |
| [PIQA](https://arxiv.org/abs/1911.11641) | Accuracy | 0-shot | 78.9 | 81.0 |
| [SocialIQA](https://arxiv.org/abs/1904.09728) | Accuracy | 0-shot | 48.8 | 50.0 |
| [TriviaQA](https://arxiv.org/abs/1705.03551) | Accuracy | 5-shot | 60.8 | 70.2 |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions) | Accuracy | 5-shot | 15.5 | 20.9 |
| [ARC-c](https://arxiv.org/abs/1911.01547) | Accuracy | 25-shot | 51.7 | 61.6 |
| [ARC-e](https://arxiv.org/abs/1911.01547) | Accuracy | 0-shot | 75.8 | 81.6 |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | Accuracy | 5-shot | 66.8 | 71.7 |
| [BIG-Bench Hard](https://paperswithcode.com/dataset/bbh) | Accuracy | few-shot | 44.3 | 52.9 |
| [DROP](https://arxiv.org/abs/1903.00161) | Token F1 score | 1-shot | 53.9 | 60.8 |

#### Multilingual

| Benchmark | Metric | n-shot | E2B IT | E4B IT |
|---|---|---|---|---|
| [MGSM](https://arxiv.org/abs/2210.03057) | Accuracy | 0-shot | 53.1 | 60.7 |
| [WMT24++](https://arxiv.org/abs/2502.12404v1) (ChrF) | Character-level F-score | 0-shot | 42.7 | 50.1 |
| [Include](https://arxiv.org/abs/2411.19799) | Accuracy | 0-shot | 38.6 | 57.2 |
| [MMLU](https://arxiv.org/abs/2009.03300) (ProX) | Accuracy | 0-shot | 8.1 | 19.9 |
| [OpenAI MMLU](https://huggingface.co/datasets/openai/MMMLU) | Accuracy | 0-shot | 22.3 | 35.6 |
| [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | Accuracy | 0-shot | 55.1 | 60.3 |
| [ECLeKTic](https://arxiv.org/abs/2502.21228) | ECLeKTic score | 0-shot | 2.5 | 1.9 |

#### STEM and code

| Benchmark | Metric | n-shot | E2B IT | E4B IT |
|---|---|---|---|---|
| [GPQA](https://arxiv.org/abs/2311.12022) Diamond | RelaxedAccuracy/accuracy | 0-shot | 24.8 | 23.7 |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) v5 | pass@1 | 0-shot | 18.6 | 25.7 |
| Codegolf v2.2 | pass@1 | 0-shot | 11.0 | 16.8 |
| [AIME 2025](https://www.vals.ai/benchmarks/aime-2025-05-09) | Accuracy | 0-shot | 6.7 | 11.6 |

#### Additional benchmarks

| Benchmark | Metric | n-shot | E2B IT | E4B IT |
|---|---|---|---|---|
| [MMLU](https://arxiv.org/abs/2009.03300) | Accuracy | 0-shot | 60.1 | 64.9 |
| [MBPP](https://arxiv.org/abs/2108.07732) | pass@1 | 3-shot | 56.6 | 63.6 |
| [HumanEval](https://arxiv.org/abs/2107.03374) | pass@1 | 0-shot | 66.5 | 75.0 |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | pass@1 | 0-shot | 13.2 | 13.2 |
| HiddenMath | Accuracy | 0-shot | 27.7 | 37.7 |
| [Global-MMLU-Lite](https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite) | Accuracy | 0-shot | 59.0 | 64.5 |
| [MMLU](https://arxiv.org/abs/2009.03300) (Pro) | Accuracy | 0-shot | 40.5 | 50.6 |

## Ethics and Safety

Ethics and safety evaluation approach and results.

### Evaluation Approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

- **Child Safety**: Evaluation of text-to-text and image to text prompts covering child safety policies, including child sexual abuse and exploitation.
- **Content Safety:** Evaluation of text-to-text and image to text prompts covering safety policies including, harassment, violence and gore, and hate speech.
- **Representational Harms**: Evaluation of text-to-text and image to text prompts covering safety policies including bias, stereotyping, and harmful associations or inaccuracies.

In addition to development level evaluations, we conduct "assurance
evaluations" which are our 'arms-length' internal evaluations for responsibility
governance decision making. They are conducted separately from the model
development team, to inform decision making about release. High level findings
are fed back to the model team, but prompt sets are held-out to prevent
overfitting and preserve the results' ability to inform decision making. Notable
assurance evaluation results are reported to our Responsibility \& Safety Council
as part of release review.

### Evaluation Results

For all areas of safety testing, we saw safe levels of performance across the
categories of child safety, content safety, and representational harms relative
to previous Gemma models. All testing was conducted without safety filters to
evaluate the model capabilities and behaviors. For text-to-text, image-to-text,
and audio-to-text, and across all model sizes, the model produced minimal policy
violations, and showed significant improvements over previous Gemma models'
performance with respect to high severity violations. A limitation of our
evaluations was they included primarily English language prompts.

## Usage and Limitations

These models have certain limitations that users should be aware of.

### Intended Usage

Open generative models have a wide range of applications across various
industries and domains. The following list of potential uses is not
comprehensive. The purpose of this list is to provide contextual information
about the possible use-cases that the model creators considered as part of model
training and development.

- Content Creation and Communication
  - **Text Generation**: Generate creative text formats such as poems, scripts, code, marketing copy, and email drafts.
  - **Chatbots and Conversational AI**: Power conversational interfaces for customer service, virtual assistants, or interactive applications.
  - **Text Summarization**: Generate concise summaries of a text corpus, research papers, or reports.
  - **Image Data Extraction**: Extract, interpret, and summarize visual data for text communications.
  - **Audio Data Extraction**: Transcribe spoken language, translate speech to text in other languages, and analyze sound-based data.
- Research and Education
  - **Natural Language Processing (NLP) and generative model
    Research**: These models can serve as a foundation for researchers to experiment with generative models and NLP techniques, develop algorithms, and contribute to the advancement of the field.
  - **Language Learning Tools**: Support interactive language learning experiences, aiding in grammar correction or providing writing practice.
  - **Knowledge Exploration**: Assist researchers in exploring large bodies of data by generating summaries or answering questions about specific topics.

### Limitations

- Training Data
  - The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.
  - The scope of the training dataset determines the subject areas the model can handle effectively.
- Context and Task Complexity
  - Models are better at tasks that can be framed with clear prompts and instructions. Open-ended or highly complex tasks might be challenging.
  - A model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).
- Language Ambiguity and Nuance
  - Natural language is inherently complex. Models might struggle to grasp subtle nuances, sarcasm, or figurative language.
- Factual Accuracy
  - Models generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.
- Common Sense
  - Models rely on statistical patterns in language. They might lack the ability to apply common sense reasoning in certain situations.

### Ethical Considerations and Risks

The development of generative models raises several ethical concerns. In
creating an open model, we have carefully considered the following:

- Bias and Fairness
  - Generative models trained on large-scale, real-world text and image data can reflect socio-cultural biases embedded in the training material. These models underwent careful scrutiny, input data pre-processing described and posterior evaluations reported in this card.
- Misinformation and Misuse
  - Generative models can be misused to generate text that is false, misleading, or harmful.
  - Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).
- Transparency and Accountability:
  - This model card summarizes details on the models' architecture, capabilities, limitations, and evaluation processes.
  - A responsibly developed open model offers the opportunity to share innovation by making generative model technology accessible to developers and researchers across the AI ecosystem.

Risks identified and mitigations:

- **Perpetuation of biases**: It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.
- **Generation of harmful content**: Mechanisms and guidelines for content safety are essential. Developers are encouraged to exercise caution and implement appropriate content safety safeguards based on their specific product policies and application use cases.
- **Misuse for malicious purposes** : Technical limitations and developer and end-user education can help mitigate against malicious applications of generative models. Educational resources and reporting mechanisms for users to flag misuse are provided. Prohibited uses of Gemma models are outlined in the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- **Privacy violations**: Models were trained on data filtered for removal of certain personal information and other sensitive data. Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.

### Benefits

At the time of release, this family of models provides high-performance open
generative model implementations designed from the ground up for responsible AI
development compared to similarly sized models.

Using the benchmark evaluation metrics described in this document, these models
have shown to provide superior performance to other, comparably-sized open model
alternatives.

There are two key decisions to make when you want to run a Gemma model:
1) what Gemma variant you want to run, and 2) what AI execution framework you
are going to use to run it? A key issue in making both these decisions has to do
with what hardware you and your users have available to run the model.

This overview helps you navigate these decisions and start working with Gemma
models. The general steps for running a Gemma model are as follows:

- [Choose a framework for running](https://ai.google.dev/gemma/docs/run#choose-a-framework)
- [Select a Gemma variant](https://ai.google.dev/gemma/docs/run#select-a-variant)
- [Run generation and inference requests](https://ai.google.dev/gemma/docs/run#run-generation)

## Choose a framework

Gemma models are compatible with a wide variety of ecosystem tools. Choosing the
right one depends on your available hardware (Cloud GPUs versus Local Laptop)
and your interface preference (Python code versus Desktop Application).

Use the following table to quickly identify the best tool for your needs:

| If you want to... | Recommended Framework | Best For |
|---|---|---|
| **Run locally with a Chat UI** | - **[LM Studio](https://ai.google.dev/gemma/docs/integrations/lmstudio)** - **[Ollama](https://ai.google.dev/gemma/docs/integrations/ollama)** | Beginners, or users who want a "Gemini-like" experience on their laptop. |
| **Run efficiently on Edge** | - **[Gemma.cpp](https://ai.google.dev/gemma/docs/core/gemma_cpp)** - **[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)** - **[llama.cpp](https://github.com/ggml-org/llama.cpp)** - **[MediaPipe LLM Inference API](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)** - **[MLX](https://github.com/ml-explore/mlx)** | High-performance local inference with minimal resources. |
| **Build/Train in Python** | - **[Gemma library for JAX](https://gemma-llm.readthedocs.io)** - **[Hugging Face Transformers](https://huggingface.co/docs/transformers/en/model_doc/gemma3)** - **[Keras](https://ai.google.dev/gemma/docs/core/keras_inference)** - **[PyTorch](https://ai.google.dev/gemma/docs/core/pytorch_gemma)** - **[Unsloth](https://unsloth.ai/blog/gemma3)** | Researchers and Developers building custom applications or fine-tuning models. |
| **Deploy to Production / Enterprise** | - **[Google Cloud Kubernetes Engine (GKE)](https://ai.google.dev/gemma/docs/core/gke)** - **[Google Cloud Run](https://ai.google.dev/gemma/docs/core/deploy_to_cloud_run_from_ai_studio)** - **[Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma)** - **[vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm)** | Scalable, managed cloud deployment with enterprise security and MLOps support. |

### Framework Details

The following are guides for running Gemma models categorized by your deployment
environment.

#### 1. Desktop \& Local Inference (High Efficiency)

These tools allow you to run Gemma on consumer hardware (laptops, desktops) by
utilizing optimized formats (like GGUF) or specific hardware accelerators.

- **[LM Studio](https://ai.google.dev/gemma/docs/integrations/lmstudio)**: A desktop application that lets you download and chat with Gemma models in a user-friendly interface. No coding required.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: A popular open-source C++ port of Llama (and Gemma) that runs incredibly fast on CPUs and Apple Silicon.
- **[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)** : Offers a command-line interface ([CLI](https://github.com/google-ai-edge/LiteRT-LM?tab=readme-ov-file#desktop-cli-lit)) to run optimized `.litertlm` Gemma models on desktop (Windows, Linux, macOS), powered by LiteRT (formerly TFLite).
- **[MLX](https://github.com/ml-explore/mlx)**: A framework designed specifically for machine learning on Apple Silicon, perfect for Mac users who want built-in performance.
- **[Gemma.cpp](https://ai.google.dev/gemma/docs/core/gemma_cpp)**: A lightweight, standalone C++ inference engine specifically from Google.
- **[Ollama](https://ai.google.dev/gemma/docs/integrations/ollama)**: A tool to run open LLMs locally, often used to power other applications.

#### 2. Python Development (Research \& Fine-tuning)

Standard frameworks for AI developers building applications, pipelines, or
training models.

- **[Hugging Face Transformers](https://huggingface.co/docs/transformers/en/model_doc/gemma3)**: The industry standard for quick access to models and pipelines.
- **[Unsloth](https://unsloth.ai/blog/gemma3)**: An optimized library for fine-tuning LLMs. It lets you train Gemma models 2-5x faster with significantly less memory, making it possible to fine-tune on consumer GPUs (e.g., free Google Colab tiers).
- **[Keras](https://ai.google.dev/gemma/docs/core/keras_inference)** / **[JAX](https://gemma-llm.readthedocs.io)** / **[PyTorch](https://ai.google.dev/gemma/docs/core/pytorch_gemma)**: Core libraries for deep learning research and custom architecture implementation.

#### 3. Mobile \& Edge Deployment (On-Device)

Frameworks designed to run LLMs directly on user devices (Android, iOS, Web)
without internet connectivity, often utilizing NPUs (Neural Processing Units).

- **[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)**: The fully open-source framework for on-device LLM development that offers maximum performance and fine-grained control, with direct support for CPU, GPU, and NPU acceleration on Android and iOS.
- **[MediaPipe LLM Inference API](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)**: The easiest way to integrate Gemma into cross-platform apps. It offers a high-level API that works across Android, iOS, and Web.

#### 4. Cloud \& Production Deployment

Managed services for scaling your application to thousands of users or accessing
massive compute power.

- **[Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma)**: Google Cloud's fully managed AI platform. Best for enterprise applications requiring SLAs and scaling.
- **[Google Cloud Kubernetes Engine (GKE)](https://ai.google.dev/gemma/docs/core/gke)**: For orchestrating your own serving clusters.
- **[vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm)**: A high-throughput and memory-efficient inference and serving engine, often used in cloud deployments.

Make sure your intended deployment Gemma model format, such as Keras built-in
format, Safetensors, or GGUF, is supported by your chosen framework.

## Select a Gemma variant

Gemma models are available in several variants and sizes, including the
foundation or [core](https://ai.google.dev/gemma/docs/core) Gemma models, and more
specialized model variants such as
[PaliGemma](https://ai.google.dev/gemma/docs/paligemma) and
[DataGemma](https://ai.google.dev/gemma/docs/datagemma), and many variants
created by the AI developer community on sites such as
[Kaggle](https://www.kaggle.com/models?query=gemma) and
[Hugging Face](https://huggingface.co/models?search=gemma). If you are unsure
about what variant you should start with, select the latest Gemma
[core](https://ai.google.dev/gemma/docs/core) instruction-tuned (IT) model with
the lowest number of parameters. This type of Gemma model has low compute
requirements and be able to respond to a wide variety of prompts without
requiring additional development.

Consider the following factors when choosing a Gemma variant:

- **Gemma core, and other variant families such as PaliGemma, CodeGemma** : *Recommend Gemma (core).* Gemma variants beyond the core version have the same architecture as the core model, and are trained to perform better at specific tasks. Unless your application or goals align with the specialization of a specific Gemma variant, it is best to start with a Gemma core, or base, model.
- **Instruction-tuned (IT), pre-trained (PT), fine-tuned (FT), mixed
  (mix)** : *Recommend IT.*
  - *Instruction-tuned* (IT) Gemma variants are models that have been trained to respond to a variety of instructions or requests in human language. These model variants are the best place to start because they can respond to prompts without further model training.
  - *Pre-trained* (PT) Gemma variants are models that have been trained to make inferences about language or other data, but have not been trained to follow human instructions. These models require additional training or tuning to be able to perform tasks effectively, and are meant for researchers or developers who want to study or develop the capabilities of the model and its architecture.
  - *Fine-tuned* (FT) Gemma variants can be considered IT variants, but are typically trained to perform a specific task, or perform well on a specific generative AI benchmark. The PaliGemma variant family includes a number of FT variants.
  - *Mixed* (mix) Gemma variants are versions of PaliGemma models that have been instruction tuned with a variety of instructions and are suitable for general use.
- **Parameters** : *Recommend smallest number available*. In general, the more parameters a model has, the more capable it is. However, running larger models requires larger and more complex compute resources, and generally slows down development of an AI application. Unless you have already determined that a smaller Gemma model cannot meet your needs, choose a one with a small number of parameters.
- **Quantization levels:** *Recommend half precision (16-bit), except for
  tuning*. Quantization is a complex topic that boils down to what size and precision of data, and consequently how much memory a generative AI model uses for calculations and generating responses. After a model is trained with high-precision data, which is typically 32-bit floating point data, models like Gemma can be modified to use lower precision data such as 16, 8 or 4-bit sizes. These quantized Gemma models can still perform well, depending on the complexity of the tasks, while using significantly less compute and memory resources. However, tools for tuning quantized models are limited and may not be available within your chosen AI development framework. Typically, you must fine-tune a model like Gemma at full precision, then quantize the resulting model.

For a list of key, Google-published Gemma models, see the
[Getting started with Gemma models](https://ai.google.dev/gemma/docs/get_started#models-list),
Gemma model list.

## Run generation and inference requests

After you have selected an AI execution framework and a Gemma variant, you can
start running the model, and prompting it to generate content or complete tasks.
For more information on how to run Gemma with a specific framework, see the
guides linked in the [Choose a framework](https://ai.google.dev/gemma/docs/run#choose-a-framework) section.

### Prompt formatting

All instruction-tuned Gemma variants have specific prompt formatting
requirements. Some of these formatting requirements are handled automatically by
the framework you use to run Gemma models, but when you are sending prompt data
directly to a tokenizer, you must add specific tags, and the tagging
requirements can change depending on the Gemma variant you are using. See the
following guides for information on Gemma variant prompt formatting and system
instructions:

- [Gemma prompt and system instructions](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [PaliGemma prompt and system instructions](https://ai.google.dev/gemma/docs/paligemma/prompt-system-instructions)
- [CodeGemma prompt and system instructions](https://ai.google.dev/gemma/docs/codegemma/prompt-structure)
- [FunctionGemma formatting and best practices](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)

# Gemma setup

This page provides setup instructions for using Gemma in Colab. Some of the
instructions are applicable to other development environments as well.

## Get access to Gemma

Before using Gemma for the first time, you must request access to the
model through Kaggle. As part of the process, you'll have to use a Kaggle
account to accept the Gemma use policy and license terms.

If you don't already have a Kaggle account, you can register for one at
[kaggle.com](https://www.kaggle.com). Then complete the following steps:

1. Go to the [Gemma model card](https://www.kaggle.com/models/google/gemma) and select **Request Access**.
2. Complete the consent form and accept the terms and conditions.

## Select a Colab runtime

To complete a Colab tutorial, you must have a Colab runtime with sufficient
resources to run the Gemma model. To [get started](https://ai.google.dev/gemma/docs/get_started), you can
use a T4 GPU:

1. In the upper-right of the Colab window, select ▾ (**Additional connection options**).
2. Select **Change runtime type**.
3. Under **Hardware accelerator** , select **T4 GPU**.

## Configure your API key

To use Gemma, you must provide your Kaggle username and a Kaggle API key. To
generate and configure these values, follow these steps:

1. To generate a Kaggle API key, go to the **Account** tab of your Kaggle [user
   profile](https://www.kaggle.com/settings) and select **Create New Token** . This will trigger the download of a `kaggle.json` file containing your API credentials.
2. Open `kaggle.json` in a text editor. The contents should look something like
   this:

       {"username":"your_username","key":"012345678abcdef012345678abcdef1a"}

3. In Colab, select **Secrets** (🔑) and add your Kaggle username and Kaggle
   API key. Store your username under the name `KAGGLE_USERNAME` and your API
   key under the name `KAGGLE_KEY`.

   | **Note:** Kaggle notebooks have a key storage feature under **Add-ons** \> **Secrets**, along with instructions for accessing stored keys.

Now you're ready to complete the remaining setup steps in Colab. If you're
working through a Colab tutorial, go to Colab and set the environment variables.
| **Tip:** As an alternative to setting environment variables, you can use `kagglehub` to [authenticate](https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate).


# Gemma formatting and system instructions

Gemma instruction-tuned (IT) models are trained with a specific *formatter* that
annotates all instruction tuning examples with extra information, both at
training and inference time. The formatter has two purposes:

1. Indicating roles in a conversation, such as the *system* , *user* , or *assistant* roles.
2. Delineating turns in a conversation, especially in a multi-turn conversation.

Below, we specify the control tokens used by Gemma and their use cases. Note
that the control tokens are reserved in and specific to our tokenizer.

- Token to indicate a user turn: `user`
- Token to indicate a model turn: `model`
- Token to indicate the beginning of dialogue turn: `<start_of_turn>`
- Token to indicate the end of dialogue turn: `<end_of_turn>`

Here's an example dialogue:  

    <start_of_turn>user
    knock knock<end_of_turn>
    <start_of_turn>model
    who is there<end_of_turn>
    <start_of_turn>user
    Gemma<end_of_turn>
    <start_of_turn>model
    Gemma who?<end_of_turn>

The token `"<end_of_turn>\n"` is the turn separator, and the prompt prefix is
`"<start_of_turn>model\n"`. This means that if you'd like to prompt the model
with a question like, "What is Cramer's Rule?", you should instead feed the
model as follows:  

    "<start_of_turn>user
    What is Cramer's Rule?<end_of_turn>
    <start_of_turn>model"

Note that if you want to finetune the pretrained Gemma models with your own
data, you can use any such schema for control tokens, as long as it's consistent
between your training and inference use cases.

## System instructions

Gemma's instruction-tuned models are designed to work with only two roles:
`user` and `model`. Therefore, the `system` role or a system turn is not
supported.

Instead of using a separate system role, provide system-level instructions
directly within the initial user prompt. The model instruction following
capabilities allow Gemma to interpret the instructions effectively. For example:  

    <start_of_turn>user
    Only reply like a pirate.

    What is the answer to life the universe and everything?<end_of_turn>
    <start_of_turn>model
    Arrr, 'tis 42,<end_of_turn>

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
