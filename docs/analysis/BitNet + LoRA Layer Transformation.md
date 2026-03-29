# **Feasibility and Implementation Plan for Layer-wise Model Transformation: Hybrid BitNet 1.58b and LoRA Residual Adapters in Bielik LLMs**

The pursuit of extreme compression in Large Language Models (LLMs) has increasingly gravitated toward low-bit quantization, representing a fundamental paradigm shift in how artificial neural networks store and process semantic representations. Among the most prominent advancements in this domain is the BitNet 1.58b architecture, which imposes a rigorous constraint on weight matrices by restricting them to a ternary alphabet. By utilizing only three discrete states, this architecture achieves profound reductions in both memory footprint and memory bandwidth requirements. Theoretically, it enables the replacement of energy-intensive floating-point matrix multiplications with highly efficient integer addition and sign inversion operations. However, applying post-training quantization to such extreme, sub-2-bit widths inherently induces significant representation collapse. This collapse manifests as acute accuracy degradation, particularly in tasks requiring zero-shot reasoning, complex instruction-following, and nuanced contextual understanding. To circumvent this severe limitation without entirely discarding the unprecedented computational benefits of ternary operations, a novel hybrid architectural paradigm has emerged at the forefront of machine learning research. This paradigm combines a strictly quantized ternary base layer with a parallel, trainable residual bridge operating in higher precision, typically parameterized via Low-Rank Adaptation (LoRA).  
This report provides an exhaustive investigation into the feasibility and the precise implementation methodology for executing a layer-wise model transformation utilizing this hybrid architecture. The specific target for this structural surgery is the Polish "Bielik" LLM family. By systematically replacing individual full-precision layers within the Bielik architecture with a BitNet-style ternary base coupled with a high-precision residual adapter, it is mathematically possible to distill the layer's original continuous latent manifold into a compressed format. Simultaneously, the residual adapter is trained to compensate for the quantization error introduced by the ternary discretization. This approach effectively bridges the persistent gap between the computational efficiency of extreme quantization and the robust expressivity inherent in full-precision, overparameterized models. The subsequent sections will synthesize the current state-of-the-art regarding ternary residual learning, evaluate the architectural suitability of the Bielik models, analyze the computational economics of generating the necessary activation datasets, detail a comprehensive PyTorch-based implementation roadmap, and critically assess the systemic risks associated with layer-wise error accumulation.

## **State-of-the-Art Search: Ternary Quantization and Residual Bridges**

The intersection of ternary quantization and residual parameter-efficient fine-tuning represents an active and highly specialized frontier in deep learning research. The fundamental premise of this intersection relies on the understanding that while an extremely quantized weight matrix captures the coarse, dominant singular vectors of the original high-dimensional weight space, it inevitably discards the fine-grained, high-frequency features necessary for nuanced language generation. The literature demonstrates that these lost features can be recovered by an auxiliary pathway.

### **The BitNet 1.58b Framework and the Straight-Through Estimator**

The BitNet 1.58b architecture employs a unique quantization mechanism distinct from traditional rounding. The weight matrices are scaled by their absolute mean and subsequently rounded to the nearest integer within the ternary domain. Because the mathematical rounding function is a step function and thus inherently non-differentiable, making standard backpropagation via automatic differentiation impossible, the architecture fundamentally relies on the Straight-Through Estimator (STE).1 The STE is a heuristic technique that bypasses the zero-gradient problem of the step function by routing the gradients from the output directly to the input during the backward pass. Essentially, it approximates the gradient of the rounding operation as an identity function, allowing the optimizer to update the underlying latent continuous weights.3  
While the STE has proven highly effective for training 1-bit and 1.58-bit models from scratch, applying it in a post-training layer-wise distillation context introduces significant optimization challenges. The gradient mismatch inherent in the STE can lead to severe instability when attempting to force a pre-trained, full-precision continuous manifold into a discrete ternary space.1 The ruggedness of the loss landscape caused by this discretization necessitates a compensatory mechanism to ensure convergence and prevent the catastrophic forgetting of the pre-trained semantic structures.

### **BitLoRA and Quantization-Compatible Adapter Tuning**

The specific synthesis of 1.58-bit ternary quantization with LoRA residual adapters has been recently formalized and empirically validated in the literature. Notably, the 2026 publication *BitLoRA: Quantization-Compatible Adapter Tuning for 1.58-bit LLM in Federated On-Device AI-Agent* provides a rigorous validation of this hybrid approach.5 The BitLoRA framework establishes that attaching a low-rank adapter to a frozen or semi-frozen 1.58-bit base allows the neural network to recover the downstream task performance that is typically lost during ternary quantization. The adapter effectively learns to model the quantization residual, acting as a localized corrective manifold that absorbs the distribution shift caused by the ternary weights.8  
Further supporting this paradigm is the development of *RA-LoRA* (Rank-Adaptive LoRA) specifically designed for 2-bit quantized LLMs.10 The RA-LoRA research demonstrates that the effectiveness of low-bit quantization-aware fine-tuning is highly dependent on rank dynamics. The method dynamically adjusts the rank of the adapter based on the mathematical sensitivity of the specific layer being quantized.10 The broader literature on residual adapters also confirms that parameterizing the residual error of a compressed layer via a parallel low-rank matrix formulation is theoretically sound.13 Techniques such as Parameter-Efficient and Quantization-aware Adaptation (PEQA) utilize a two-stage pipeline where the pre-trained feed-forward network weight matrices are quantized, and a residual adapter is subsequently tuned to regain the lost precision.13 This convergence of research unequivocally confirms that a BitNet 1.58b base combined with a trainable residual bridge is a viable, state-of-the-art methodology for extreme layer-wise model compression.

## **Architectural Evaluation and Model Selection: The Bielik Family**

The "Bielik" family of Large Language Models, developed collaboratively by SpeakLeash and ACK Cyfronet AGH, represents the current state-of-the-art in Polish natural language processing. The models have been released in various iterations and parameter scales, predominantly the 7B variants (such as Bielik-7B-v2 and Bielik-7B-Instruct-v0.1) and the 11B variants (such as Bielik-11B-v2.3-Instruct).15

### **Structural Analysis of Bielik-11B-v2.3-Instruct**

For the execution of a highly complex structural surgery such as layer-wise ternary substitution, the target model's architectural stability, parameter distribution, and layer redundancy are of paramount importance. Based on a comprehensive architectural assessment, Bielik-11B-v2.3-Instruct emerges as the superior candidate for this transformation when compared to the 7B variants.  
Bielik-11B-v2.3 is mathematically formulated as a linear merge of the v2.0, v2.1, and v2.2 instruct models. It is fundamentally built upon the Mistral 7B v0.2 architecture and was scaled to 11 billion parameters utilizing an architectural technique known as "depth up-scaling".15 The precise architectural specifications of this model include a hidden dimension size of 4096, an intermediate (MLP) expansion size of 14336, 32 query attention heads, and 8 Key/Value heads implementing Grouped-Query Attention (GQA), all functioning utilizing the SwiGLU non-linear activation function.19

| Architectural Parameter | Bielik-11B-v2.3 Specification | Implications for Layer-wise Transformation |
| :---- | :---- | :---- |
| **Model Dimension (Hidden Size)** | 4096 | Requires a LoRA adapter rank capable of capturing a 4096-dimensional residual space. |
| **Intermediate Size (MLP)** | 14336 | Represents the largest parameter cluster per layer; primary target for ternary compression. |
| **Attention Heads (Q / KV)** | 32 / 8 (GQA) | Ternary quantization of Key/Value projections requires careful attention to avoid context decay. |
| **Activation Function** | SwiGLU | Non-linear amplification of upstream quantization errors necessitates rigorous layer-wise error mitigation. |

The critical factor rendering Bielik-11B-v2.3 optimal for this specific experiment is its depth up-scaling origin. Depth up-scaling typically involves duplicating or systematically interpolating layers from a smaller base model to create a deeper, more expressive network. Consequently, models generated via depth up-scaling exhibit a significantly higher degree of functional similarity and representation overlap between adjacent layers compared to networks trained natively from scratch. This inherent layer redundancy acts as a topological buffer, making the network highly resilient to the minor representational shocks introduced by swapping a pristine full-precision layer with a hybrid ternary-residual layer.21 The continuous, highly correlated transition of hidden states across the up-scaled layers minimizes the risk of catastrophic representation collapse during the hot-swapping phase. Furthermore, the model weights for Bielik-11B-v2.3-Instruct are fully open-source and readily available in safe tensor formats on the Hugging Face hub, enabling immediate access for localized structural surgery without licensing restrictions.22

## **Inference Cost Analysis and Activation Dataset Logistics**

To perform layer-wise distillation effectively, it is mandatory to compile a comprehensive dataset consisting of input and output activations for the specific layer undergoing the transformation. The objective outlined is to generate a dataset of 5,000 activation samples per layer for the 11B model, which will serve as the ground-truth manifold for the distillation loss function.

### **Storage and Memory Footprint Estimation**

Operating on the Bielik-11B-v2.3 architecture, the hidden representation size is 4096\. Assuming a standard calibration sequence length of 2048 tokens and a batch size of 1, the memory footprint of a single activation tensor in BF16 (bfloat16) precision can be calculated as follows:

$$\\text{Tensor Size} \= 1 \\text{ (batch)} \\times 2048 \\text{ (tokens)} \\times 4096 \\text{ (dimensions)} \\times 2 \\text{ (bytes per BF16 value)}$$

$$\\text{Tensor Size} \= 16,777,216 \\text{ bytes} \\approx 16.77 \\text{ Megabytes (MB)}$$  
For a dataset comprising 5,000 unique forward pass samples, the storage requirement for the input activations of a single layer is approximately 83.88 Gigabytes (GB). To execute the distillation training, both the input activations (required to feed the new hybrid layer) and the teacher's original output activations (required to compute the target loss) must be cached simultaneously. Therefore, the absolute total storage requirement per single layer is roughly 167.76 GB. Given that the depth up-scaled 11B model contains a high number of layers (approximately 48 layers), attempting to store the entire model's activation dataset on disk simultaneously would require over 8 Terabytes (TB) of high-speed NVMe storage. This presents a massive I/O bottleneck.

### **Cloud versus Local Compute Economics**

Generating these activations requires running 5,000 forward passes of the full-precision Bielik-11B-v2.3 model. The 11 billion parameters in FP16/BF16 consume approximately 22 GB of VRAM during inference 24, making it entirely feasible to run on high-end consumer-grade GPUs equipped with 24GB of VRAM, such as the NVIDIA RTX 3090 or RTX 4090\. The following analysis compares the costs and logistical realities of generating this dataset and conducting the subsequent distillation training across different hardware provisions.

| Infrastructure Provider | Hardware Specification | Estimated Cost per Hour | Logistical and Performance Considerations |
| :---- | :---- | :---- | :---- |
| **Vast.ai** (P2P Cloud) | 1x NVIDIA RTX 3090 (24GB) | $0.14 \- $0.16 25 | Highly cost-effective. However, host machine disk I/O variability can bottleneck the continuous writing of 16MB tensors. Extra costs apply for large persistent storage volumes. |
| **Vast.ai** (P2P Cloud) | 1x NVIDIA RTX 4090 (24GB) | $0.16 \- $0.32 26 | Superior compute speed for generating forward passes quickly. Subject to the same peer-to-peer network reliability risks as the 3090 instances. |
| **RunPod** (Datacenter) | 1x NVIDIA RTX 3090 (24GB) | $0.22 25 | Datacenter-grade network storage mounts (network volumes) provide consistent bandwidth necessary for moving 100GB+ datasets without I/O stalling. |
| **RunPod** (Datacenter) | 1x NVIDIA RTX 4090 (24GB) | $0.20 \- $0.69 26 | Premium tier offers enterprise reliability. The optimal choice for uninterrupted pipeline execution, though network storage accumulation costs must be monitored. |
| **Local Hardware** | 1x NVIDIA RTX 3090/4090 | Sunk Capital Cost \+ Power | Zero egress costs. Permits unlimited utilization of local PCIe Gen4/Gen5 NVMe storage arrays. Requires a high upfront capital expenditure of $1,500 to $2,500.27 |

When operating in a cloud environment like Vast.ai or RunPod, generating and storing 167 GB of data per layer becomes an IOPS-bound operation rather than a compute-bound one. While cloud storage costs themselves are relatively modest—typically around $17 per month for a few Terabytes on S3 or equivalent block storage volumes 29—peer-to-peer providers may possess severe network bottlenecks if the host machine's physical disk I/O cannot sustain the continuous, high-speed writes of the activation tensors.  
To navigate this logistical hurdle, the optimal approach is the implementation of an **ephemeral data pipeline**. Instead of attempting to store 8 TB of activations for the entire model prior to training, the pipeline must dynamically generate and cache the 5,000 input/output activations exclusively for *Layer N*. Once cached, the system trains the hybrid *Layer N* adapter, executes the weight hot-swap, deletes the 167 GB cache from the disk, and only then proceeds to generate the activations for *Layer N+1*. This rolling-window methodology restricts the maximum storage requirement to roughly 200 GB at any given time, vastly reducing overhead and enabling the use of highly affordable, low-storage cloud instances.

## **Detailed Implementation Plan and PyTorch Integration**

The execution of the layer-wise transformation requires a meticulously orchestrated, stage-by-line pipeline. The process must seamlessly transition from the initial data collection hook registration to the precise PyTorch class definition, distillation training execution, and finally, the network weight hot-swapping.

### **Phase 1: Activation Capture and Dataset Generation**

To successfully distill a layer, the training loop requires the input hidden states ($X$) entering the target layer and the ground-truth output hidden states ($Y$) exiting that exact same layer.

1. Initialize and load the pre-trained Bielik-11B-v2.3 model in BF16 precision onto the target GPU to prevent precision-based numerical overflow.  
2. Register PyTorch forward hooks (register\_forward\_hook) specifically on the target MistralDecoderLayer undergoing the transformation.  
3. Execute a forward pass loop feeding 5,000 diverse, high-quality Polish text sequences through the model. To capture the full linguistic breadth of the Bielik model, datasets such as CulturaX-PL should be utilized for the calibration inputs.24  
4. Serialize and save the captured $(X, Y)$ tensor pairs to local NVMe storage. Utilizing a memory-mapped format like HDF5 or SafeTensors is critical to bypass system RAM bottlenecks during the subsequent dataloading phase of training.

### **Phase 2: PyTorch Hybrid Layer Architecture**

The fundamental core of this architectural transformation is the custom PyTorch module. It explicitly requires a straight-through estimator to handle the ternary quantization of the base weights during the backward pass, combined with a parallel LoRA bypass operating in BF16. The implementation directly leverages the torch.autograd.Function class to define the custom STE mechanics.1

Python

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import math

class TernaryQuantizeSTE(torch.autograd.Function):  
    """  
    Implements the Straight-Through Estimator for BitNet 1.58b ternary quantization.  
    The forward pass discretizes the continuous weights to {-1, 0, 1}.  
    The backward pass approximates the gradient as an identity function.  
    """  
    @staticmethod  
    def forward(ctx, weight):  
        \# BitNet 1.58b standard: scale by the absolute mean of the weight tensor  
        scale \= weight.abs().mean().clamp(min=1e-8)  
          
        \# Scale the weights and apply deterministic rounding to {-1, 0, 1}  
        quantized\_weight \= torch.round(weight / scale).clamp(-1, 1\)  
          
        \# Save the scale factor to maintain variance alignment during the forward pass  
        ctx.save\_for\_backward(weight)  
          
        \# Denormalize the quantized weights for the linear operation  
        return quantized\_weight \* scale

    @staticmethod  
    def backward(ctx, grad\_output):  
        \# The Straight-Through Estimator: pass gradients directly through the rounding step  
        return grad\_output

class HybridTernaryResidualLayer(nn.Module):  
    """  
    A hybrid linear layer replacing standard nn.Linear.  
    Combines a ternary quantized base weight with a high-precision LoRA residual adapter.  
    """  
    def \_\_init\_\_(self, in\_features, out\_features, rank=32, lora\_alpha=64):  
        super().\_\_init\_\_()  
        self.in\_features \= in\_features  
        self.out\_features \= out\_features  
          
        \# Base weight parameter. In a production pipeline, this is initialized directly   
        \# from the pre-trained full-precision Bielik model weights.  
        self.weight \= nn.Parameter(torch.empty(out\_features, in\_features))  
          
        \# LoRA Residual Adapter parameters operating in BF16  
        self.lora\_A \= nn.Parameter(torch.zeros(rank, in\_features, dtype=torch.bfloat16))  
        self.lora\_B \= nn.Parameter(torch.zeros(out\_features, rank, dtype=torch.bfloat16))  
          
        \# Standard LoRA scaling factor  
        self.scaling \= lora\_alpha / rank  
          
        self.reset\_parameters()

    def reset\_parameters(self):  
        \# Initialize LoRA A with Kaiming uniform to establish an active initial subspace  
        nn.init.kaiming\_uniform\_(self.lora\_A, a=math.sqrt(5))  
        \# Initialize LoRA B with zeros so the initial adapter contribution is exactly zero  
        nn.init.zeros\_(self.lora\_B)

    def forward(self, x):  
        \# Pathway 1: Ternary Base utilizing the Straight-Through Estimator  
        w\_ternary \= TernaryQuantizeSTE.apply(self.weight)  
        base\_out \= F.linear(x, w\_ternary)  
          
        \# Pathway 2: Residual LoRA Adapter evaluated in high precision  
        lora\_out \= F.linear(F.linear(x, self.lora\_A), self.lora\_B) \* self.scaling  
          
        \# Integration: The hybrid output is the summation of the ternary base and the residual correction  
        return base\_out \+ lora\_out

### **Phase 3: Model Sub-Component Substitution and Hot-Swapping**

1. **Extraction and Initialization:** For a specific target layer (e.g., the query projection q\_proj), extract the pre-trained full-precision weights from the Hugging Face MistralForCausalLM object. Inject these weights directly into the self.weight parameter of the newly instantiated HybridTernaryResidualLayer.  
2. **Distillation Training:** Utilizing the captured 167 GB activation dataset, execute the training loop to optimize the lora\_A, lora\_B, and (under a highly constrained learning rate) the continuous latent self.weight.  
3. **Hot-Swapping:** Post-convergence, physically replace the original nn.Linear module in the model's architecture graph with the trained HybridTernaryResidualLayer object.  
4. **Iteration:** Delete the localized activation cache, increment the layer target, and repeat the pipeline sequentially from depth 0 to the final layer.

## **Loss Function Strategy: The Necessity of Mean Squared Error**

When defining the distillation objective for this layer-wise transformation, it is critical to distinguish between end-to-end model distillation and intermediate representation distillation. While Kullback-Leibler (KL) Divergence is the industry standard for matching output probability distributions (logits) between a teacher and a student model, applying KL divergence directly to intermediate hidden states is mathematically invalid. Hidden states within the intermediate layers of the Bielik model do not form normalized probability distributions; they are continuous, unconstrained vectors existing within a high-dimensional latent space.  
Therefore, the distillation objective must be formulated using Mean Squared Error (MSE) loss applied directly to the hidden states.30 As established in recent literature analyzing 1.58-bit distillation architectures, minimizing the MSE between intermediate representations is fundamentally vital for capturing the intricate, non-linear complexities that are entirely missed by the primary ternary approximation.32 The MSE objective aggressively forces the output manifold of the hybrid layer to conform to the exact topological shape of the original full-precision layer's output.  
The loss function for a given layer $l$ must be defined as follows:

$$\\mathcal{L}\_{layer} \= \\frac{1}{N} \\sum\_{i=1}^{N} \\left\\| H\_{T}^{(l)}(X\_i) \- \\left( \\text{BitLinear}(X\_i) \+ \\text{LoRA}(X\_i) \\right) \\right\\|\_2^2$$  
Where $H\_{T}^{(l)}$ represents the target ground-truth activation generated by the original Bielik layer, and $X\_i$ represents the input activation fed into both the teacher and the hybrid student. Furthermore, relying exclusively on MSE for hidden states prevents the directional scale blindness that angular metrics might suffer from. However, literature suggests that if the magnitude of the quantized vectors begins to collapse, a joint metric combining MSE (to penalize magnitude gaps) and Cosine Similarity (to penalize directional, angular biases) yields superior optimization stability.34

## **Training Dynamics: Differential Learning Rates and Warm Starts**

Training a hybrid architecture introduces highly complex optimization dynamics, primarily arising from the disparate nature of the architectural components involved. The system is attempting to simultaneously optimize a dense, fully pre-trained base matrix undergoing non-differentiable quantization via the STE, alongside a sparse, high-precision adapter initialized near zero. Successfully navigating this specific "warm start" scenario requires the strict implementation of Differential Learning Rates.35

### **The Necessity of Asymmetric Optimization**

In the hybrid layer configuration, the base weight matrix $W$ already contains highly structured, deeply optimized pre-trained knowledge from the original Bielik-11B-v2.3 model. If the learning rate applied to the latent continuous weights of $W$ is set too high, the gradients flowing back through the STE will indiscriminately alter and destroy these pre-trained semantic structures, resulting in rapid and irreversible catastrophic forgetting. Conversely, the LoRA matrices ($A$ and $B$) face a fundamentally different optimization landscape. They are initialized randomly (Kaiming uniform) and with absolute zeros, respectively. This means they possess zero pre-trained knowledge and must learn the residual error mapping entirely from scratch.38  
To mathematically reconcile this discrepancy, a differential learning rate strategy is absolutely mandatory.

| Component | Target Learning Rate | Optimization Rationale |
| :---- | :---- | :---- |
| **Quantized Base ($W$)** | Frozen or $\\approx 1 \\times 10^{-6}$ | Must act as a rigid structural anchor during initial epochs. Extremely low learning rate preserves the pre-trained Mistral architecture semantics.35 |
| **LoRA Residual ($A$, $B$)** | $\\approx 1 \\times 10^{-4}$ | Requires a significantly higher magnitude learning rate to rapidly escape the zero-initialization state and establish a meaningful corrective subspace.38 |

### **Addressing Gradient Flow Asymmetry in LoRA**

A critical nuance in training LoRA adapters alongside quantized bases is the inherent gradient flow asymmetry present at initialization. Because the matrix $B$ is explicitly initialized to zero, the initial forward pass yields a zero contribution from the adapter module. During the very first backward pass, only matrix $B$ receives meaningful, actionable gradients. Matrix $A$, however, receives near-zero gradients because its gradient calculation is dependent on the weights of $B$ ($\\nabla A \= B^T \\nabla L \\approx 0$).  
Recent studies focusing on high-performance LLM fine-tuning frameworks, such as the *Chronicals* architecture and theoretical derivations of the LoRA+ methodology, demonstrate that applying a uniform learning rate to both the $A$ and $B$ matrices results in severely sub-optimal feature learning and delayed convergence.38 To artificially accelerate convergence and maximize the expressiveness of the BF16 adapter during the distillation phase, an intra-adapter differential learning rate ratio of $\\eta\_B \= 16\\eta\_A$ should be employed.40 This specific scalar ratio allows the $B$ matrix to rapidly ascend from the zero-initialization saddle point, aggressively projecting the adapter into a useful subspace required to correct the ternary quantization errors effectively.

## **Risk Assessment: Layer-wise Error Accumulation**

While the layer-wise distillation protocol is vastly more computationally efficient and minimizes memory overhead compared to full end-to-end model distillation, it introduces the most severe structural vulnerability in this entire methodology: the phenomenon of Error Accumulation.

### **The Mechanics of Reconstruction Error Explosion**

When an individual layer $L\_n$ is successfully distilled and swapped with its hybrid ternary-residual counterpart, it inherently introduces a marginal, non-zero reconstruction error $\\epsilon\_n$, despite the rigorous MSE optimization applied during training. Evaluated in isolated layer tests, this $\\epsilon\_n$ appears mathematically negligible. However, as the forward pass continues sequentially through the deep network, the slightly perturbed output hidden state $H\_n \+ \\epsilon\_n$ serves as the direct input to the subsequent layer $L\_{n+1}$. Because layer $L\_{n+1}$ was originally trained in the pre-training phase on the pristine, unperturbed hidden states $H\_n$ and not on the newly introduced noisy manifold, the error is non-linearly amplified. This amplification is driven by the SwiGLU non-linear activation functions and the complex normalization mechanics of the attention layers.21  
This compounding effect is mathematically described in recent literature as a "reconstruction error explosion" 42 or "compounding denoising errors." It fundamentally shifts the layer-wise attention error distribution across the depth of the model.43 Over the approximately 48 layers of the depth up-scaled Bielik-11B-v2.3 model, this sequential topological misalignment causes the dominant noise component to aggressively eclipse the ideal output state. The ultimate result is catastrophic token flipping during the autoregressive decoding phase, where the model begins to hallucinate or output incoherent text despite each individual layer demonstrating low MSE during its respective distillation phase.45

### **Mitigation Strategies and Topological Interventions**

To safeguard the structural integrity and linguistic coherence of the transformed Bielik model, the following robust mitigation strategies must be integrated directly into the implementation roadmap:

1. **Quantization Error Propagation (QEP):** Relying strictly on layer-wise independent post-training distillation is proven to be insufficient for deep networks. The QEP framework mitigates accumulation by explicitly propagating the quantization errors from layer $n$ into the distillation objective of layer $n+1$.47 In operational practice, this dictates that the input activation dataset generated for Layer $n+1$ must be produced using the outputs generated by the *already swapped and quantized* hybrid Layer $n$, rather than using the original teacher's pristine, unquantized activations. This progressive, sequential data pipeline forces the subsequent LoRA adapters deeper in the network to implicitly learn to absorb and correct upstream ternary quantization errors as part of their optimization objective.50  
2. **Block-wise vs. Layer-wise Reconstruction:** If the sequential layer-wise error remains unmanageable despite QEP, the distillation granularity must be fundamentally expanded from layer-wise to block-wise reconstruction.51 Instead of optimizing a single MistralDecoderLayer at a time, the model architecture is partitioned into functional blocks comprising 2 to 4 consecutive layers. The MSE loss objective is then computed exclusively at the output of the entire block. This technique avoids the unreliable, high-variance gradients of single-layer distillation and naturally suppresses intra-block error accumulation by allowing the internal layers to co-adapt to the ternary constraints.41  
3. **Adaptive Rank Allocation (MOO):** Drawing heavily from the mathematical principles of RA-LoRA and Multi-Objective Optimization (MOO) frameworks 10, the architectural rank of the residual adapter must not be static across the network depth. Layers that exhibit historically high sensitivity to ternary quantization—typically the earliest layers responsible for foundational syntactic feature extraction and the final layers mapping semantic vectors to the vocabulary space—must be allocated a significantly higher LoRA rank (e.g., $r=64$ or $r=128$). Conversely, intermediate layers, which benefit the most from the structural redundancy created by Bielik's depth up-scaling process, can be assigned much lower ranks (e.g., $r=16$) to maximize parameter efficiency without sacrificing global accuracy.  
4. **Re-normalization Anchors:** The implementation should consider inserting intermittent synchronization steps or "anchors" within the network depth where the hybrid layer outputs are explicitly re-normalized against the original full-precision statistics. This technique forcibly resets the trajectory of the hidden state manifold, preventing the second-order cross-terms of the accumulated error from mathematically dominating the representation space before they reach the final output heads.21

The integration of these specific mitigation strategies transforms the layer-wise substitution process from a fragile, error-prone procedure into a highly robust, mathematically grounded methodology for extreme LLM compression. By systematically addressing the gradient pathologies of the STE, managing the asymmetric optimization of the residual adapters, and mathematically neutralizing the reconstruction error explosion, the proposed hybrid architecture provides a highly viable path forward for deploying massive, language-specific models like Bielik on heavily constrained edge hardware.

#### **Cytowane prace**

1. How to Fine-tune LLMs to 1.58 bits? \- Analytics Vidhya, otwierano: marca 26, 2026, [https://www.analyticsvidhya.com/blog/2024/10/fine-tune-llms-to-1-58-bits/](https://www.analyticsvidhya.com/blog/2024/10/fine-tune-llms-to-1-58-bits/)  
2. ViT-1.58b: Mobile Vision Transformers in the 1-bit Era \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2406.18051v1](https://arxiv.org/html/2406.18051v1)  
3. BitNet Training: Advanced QAT Infrastructure \- Lib.rs, otwierano: marca 26, 2026, [https://lib.rs/crates/bitnet-training](https://lib.rs/crates/bitnet-training)  
4. cross-layer design to enhance the functionality and robustness of computing-in-memory for deep neural network accelerators \- Purdue University Graduate School, otwierano: marca 26, 2026, [https://hammer.purdue.edu/ndownloader/files/55684874](https://hammer.purdue.edu/ndownloader/files/55684874)  
5. RA-LoRA: Rank-Adaptive Parameter-Efficient Fine-Tuning for Accurate 2-bit Quantized Large Language Models | Request PDF \- ResearchGate, otwierano: marca 26, 2026, [https://www.researchgate.net/publication/384216924\_RA-LoRA\_Rank-Adaptive\_Parameter-Efficient\_Fine-Tuning\_for\_Accurate\_2-bit\_Quantized\_Large\_Language\_Models](https://www.researchgate.net/publication/384216924_RA-LoRA_Rank-Adaptive_Parameter-Efficient_Fine-Tuning_for_Accurate_2-bit_Quantized_Large_Language_Models)  
6. KangYoon Lee (0000-0003-3078-6166) \- ORCID, otwierano: marca 26, 2026, [https://orcid.org/0000-0003-3078-6166](https://orcid.org/0000-0003-3078-6166)  
7. BitLoRA: Quantization-Compatible Adapter Tuning for 1.58-bit LLM in Federated On-Device AI-Agent | Request PDF \- ResearchGate, otwierano: marca 26, 2026, [https://www.researchgate.net/publication/400293673\_BitLoRA\_Quantization-Compatible\_Adapter\_Tuning\_for\_158-bit\_LLM\_in\_Federated\_On-Device\_AI-Agent](https://www.researchgate.net/publication/400293673_BitLoRA_Quantization-Compatible_Adapter_Tuning_for_158-bit_LLM_in_Federated_On-Device_AI-Agent)  
8. Bit-LoRA as an application of BitNet and 1.58 bit neural network technologies \- Medium, otwierano: marca 26, 2026, [https://medium.com/data-science/bit-lora-as-an-application-of-bitnet-and-1-58-bit-neural-network-technologies-17ee80bf79f9](https://medium.com/data-science/bit-lora-as-an-application-of-bitnet-and-1-58-bit-neural-network-technologies-17ee80bf79f9)  
9. FedBiOT: LLM Local Fine-tuning in Federated Learning without Full Model \- ResearchGate, otwierano: marca 26, 2026, [https://www.researchgate.net/publication/383419623\_FedBiOT\_LLM\_Local\_Fine-tuning\_in\_Federated\_Learning\_without\_Full\_Model](https://www.researchgate.net/publication/383419623_FedBiOT_LLM_Local_Fine-tuning_in_Federated_Learning_without_Full_Model)  
10. RA-LoRA: Rank-Adaptive Parameter-Efficient Fine-Tuning for Accurate 2-bit Quantized Large Language Models \- ACL Anthology, otwierano: marca 26, 2026, [https://aclanthology.org/2024.findings-acl.933/](https://aclanthology.org/2024.findings-acl.933/)  
11. RA-LoRA: Rank-Adaptive Parameter-Efficient Fine-Tuning for Accurate 2-bit Quantized Large Language Models \- ACL Anthology, otwierano: marca 26, 2026, [https://aclanthology.org/2024.findings-acl.933.pdf](https://aclanthology.org/2024.findings-acl.933.pdf)  
12. \[PDF\] RA-LoRA: Rank-Adaptive Parameter-Efficient Fine-Tuning for Accurate 2-bit Quantized Large Language Models | Semantic Scholar, otwierano: marca 26, 2026, [https://www.semanticscholar.org/paper/8a2f533154e3d76661bb29897678105189751f53](https://www.semanticscholar.org/paper/8a2f533154e3d76661bb29897678105189751f53)  
13. Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2403.14608v1](https://arxiv.org/html/2403.14608v1)  
14. Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey \- OpenReview, otwierano: marca 26, 2026, [https://openreview.net/pdf?id=lIsCS8b6zj](https://openreview.net/pdf?id=lIsCS8b6zj)  
15. speakleash/Bielik-11B-v2.3-Instruct \- Hugging Face, otwierano: marca 26, 2026, [https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct)  
16. speakleash/Bielik-7B-v0.1 \- Hugging Face, otwierano: marca 26, 2026, [https://huggingface.co/speakleash/Bielik-7B-v0.1](https://huggingface.co/speakleash/Bielik-7B-v0.1)  
17. Bielik 11B v2 Technical Report \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/pdf/2505.02410](https://arxiv.org/pdf/2505.02410)  
18. Bielik 11B v2 Technical Report \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2505.02410v1](https://arxiv.org/html/2505.02410v1)  
19. config.json · deepseek-ai/DeepSeek-Prover-V2-7B at main \- Hugging Face, otwierano: marca 26, 2026, [https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B/blob/main/config.json](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B/blob/main/config.json)  
20. Bielik 11B v2 Technical Report \- arXiv.org, otwierano: marca 26, 2026, [https://arxiv.org/html/2505.02410v2](https://arxiv.org/html/2505.02410v2)  
21. Leveraging the True Depth of LLMs, otwierano: marca 26, 2026, [https://arxiv.org/html/2502.02790v3](https://arxiv.org/html/2502.02790v3)  
22. speakleash/Bielik-11B-v2.3-Instruct at main \- Hugging Face, otwierano: marca 26, 2026, [https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct/tree/main](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct/tree/main)  
23. speakleash/Bielik-11B-v2.6-Instruct \- Hugging Face, otwierano: marca 26, 2026, [https://huggingface.co/speakleash/Bielik-11B-v2.6-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.6-Instruct)  
24. Bielik-Q2-Sharp: A Comparative Study of Extreme 2-bit Quantization Methods for a Polish 11B Language Model \- arXiv.org, otwierano: marca 26, 2026, [https://arxiv.org/html/2603.04162v1](https://arxiv.org/html/2603.04162v1)  
25. RTX 3090 Cloud Pricing: Compare 5+ Providers (2026) \- GetDeploying, otwierano: marca 26, 2026, [https://getdeploying.com/gpus/nvidia-rtx-3090](https://getdeploying.com/gpus/nvidia-rtx-3090)  
26. Salad Slashes GPU Pricing Again – Now the Serving the Lowest Cost GPUs on the Market, otwierano: marca 26, 2026, [https://blog.salad.com/lowest-cost-gpus/](https://blog.salad.com/lowest-cost-gpus/)  
27. The real cost of hosting an LLM : r/LocalLLaMA \- Reddit, otwierano: marca 26, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1jzeo0l/the\_real\_cost\_of\_hosting\_an\_llm/](https://www.reddit.com/r/LocalLLaMA/comments/1jzeo0l/the_real_cost_of_hosting_an_llm/)  
28. LLM Implementation and Maintenance Costs for Businesses: A Detailed Breakdown, otwierano: marca 26, 2026, [https://inero-software.com/llm-implementation-and-maintenance-costs-for-businesses-a-detailed-breakdown/](https://inero-software.com/llm-implementation-and-maintenance-costs-for-businesses-a-detailed-breakdown/)  
29. The Economics of Deploying Large Language Models: Costs, Value, and a 99.7% Savings Story \- Cloudurable, otwierano: marca 26, 2026, [https://cloudurable.com/blog/the-economics-of-deploying-large-language-models-costs-value-and-a-997-savings-story/](https://cloudurable.com/blog/the-economics-of-deploying-large-language-models-costs-value-and-a-997-savings-story/)  
30. Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2601.22709v1](https://arxiv.org/html/2601.22709v1)  
31. Mixture of Scales: Memory-Efficient Token-Adaptive Binarization for Large Language Models \- NIPS papers, otwierano: marca 26, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/f89221edad5a6a4a54fcf247cb37cd62-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/f89221edad5a6a4a54fcf247cb37cd62-Paper-Conference.pdf)  
32. LittleBit: Ultra Low-Bit Quantization via Latent Factorization \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2506.13771v2](https://arxiv.org/html/2506.13771v2)  
33. OneBit: Towards Extremely Low-bit Large Language Models \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2402.11295v3](https://arxiv.org/html/2402.11295v3)  
34. PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models \- ACL Anthology, otwierano: marca 26, 2026, [https://aclanthology.org/2025.acl-long.225.pdf](https://aclanthology.org/2025.acl-long.225.pdf)  
35. Differential Learning Rates \- Slav, otwierano: marca 26, 2026, [https://blog.slavv.com/differential-learning-rates-59eff5209a4f](https://blog.slavv.com/differential-learning-rates-59eff5209a4f)  
36. Awesome Generative Ai Guide | PDF | Artificial Intelligence \- Scribd, otwierano: marca 26, 2026, [https://www.scribd.com/document/976467040/Awesome-Generative-Ai-Guide](https://www.scribd.com/document/976467040/Awesome-Generative-Ai-Guide)  
37. Practical Machine Learning for Computer Vision: End-to-End Machine Learning for Images \[1 ed.\] 1098102363, 9781098102364 \- DOKUMEN.PUB, otwierano: marca 26, 2026, [https://dokumen.pub/practical-machine-learning-for-computer-vision-end-to-end-machine-learning-for-images-1nbsped-1098102363-9781098102364.html](https://dokumen.pub/practical-machine-learning-for-computer-vision-end-to-end-machine-learning-for-images-1nbsped-1098102363-9781098102364.html)  
38. Chronicals: A High-Performance Framework for LLM Fine-Tuning with 3.51x Speedup over Unsloth \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2601.02609v1](https://arxiv.org/html/2601.02609v1)  
39. A 2 \-LLM: An End-to-end Conversational Audio Avatar Large Language Model \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2602.04913v1](https://arxiv.org/html/2602.04913v1)  
40. (PDF) Chronicals: A High-Performance Framework for LLM Fine-Tuning with 3.51x Speedup over Unsloth \- ResearchGate, otwierano: marca 26, 2026, [https://www.researchgate.net/publication/399522458\_Chronicals\_A\_High-Performance\_Framework\_for\_LLM\_Fine-Tuning\_with\_351x\_Speedup\_over\_Unsloth](https://www.researchgate.net/publication/399522458_Chronicals_A_High-Performance_Framework_for_LLM_Fine-Tuning_with_351x_Speedup_over_Unsloth)  
41. LLM-BIP: Structured Pruning for Large Language Models with Block-Wise Forward Importance Propagation \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2412.06419v1](https://arxiv.org/html/2412.06419v1)  
42. ICML Poster Determining Layer-wise Sparsity for Large Language Models Through a Theoretical Perspective, otwierano: marca 26, 2026, [https://icml.cc/virtual/2025/poster/44037](https://icml.cc/virtual/2025/poster/44037)  
43. ICML Poster KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference, otwierano: marca 26, 2026, [https://icml.cc/virtual/2025/poster/43487](https://icml.cc/virtual/2025/poster/43487)  
44. KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference \- GitHub, otwierano: marca 26, 2026, [https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25dd/li25dd.pdf](https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25dd/li25dd.pdf)  
45. KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2502.04420v5](https://arxiv.org/html/2502.04420v5)  
46. GRIFFIN: Effective Token Alignment for Faster Speculative Decoding \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2502.11018v3](https://arxiv.org/html/2502.11018v3)  
47. Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization \- OpenReview, otwierano: marca 26, 2026, [https://openreview.net/pdf?id=a3l3K9khbL](https://openreview.net/pdf?id=a3l3K9khbL)  
48. Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2504.09629v2](https://arxiv.org/html/2504.09629v2)  
49. Quantization Error Propagation: Revisiting Layer-Wise Post-Training... \- OpenReview, otwierano: marca 26, 2026, [https://openreview.net/forum?id=a3l3K9khbL\&referrer=%5Bthe%20profile%20of%20Yuma%20Ichikawa%5D(%2Fprofile%3Fid%3D\~Yuma\_Ichikawa1)](https://openreview.net/forum?id=a3l3K9khbL&referrer=%5Bthe+profile+of+Yuma+Ichikawa%5D\(/profile?id%3D~Yuma_Ichikawa1\))  
50. Track: Poster Session 6 East \- ICML 2026, otwierano: marca 26, 2026, [https://icml.cc/virtual/2025/session/50268](https://icml.cc/virtual/2025/session/50268)  
51. CBQ: Cross-Block Quantization for Large Language Models \- arXiv, otwierano: marca 26, 2026, [https://arxiv.org/html/2312.07950v3](https://arxiv.org/html/2312.07950v3)  
52. What is LLM Quantization Understanding Its Importance and Techniques \- Cognativ, otwierano: marca 26, 2026, [https://www.cognativ.com/blogs/post/what-is-llm-quantization-understanding-its-importance-and-techniques/321](https://www.cognativ.com/blogs/post/what-is-llm-quantization-understanding-its-importance-and-techniques/321)