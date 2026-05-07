# CG-MLLM

**[ICML 2026] CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models**

Junming Huang, Chi Wang, Letian Li, Guangkai Xu, Donglin Huang, Hao Chen, Qiang Dai, Weiwei Xu

[[Paper](https://arxiv.org/abs/2601.21798)] [[PDF](https://arxiv.org/pdf/2601.21798)] [Project Page: Coming soon] [Code: Coming soon] [Checkpoints: Coming soon]

## News

- **2026-05-07**: 祝贺我们的论文 **CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models** 中稿 ICML 2026！我们会和大家在首尔见面。

## Overview

**CG-MLLM** is a unified multimodal large language model for 3D captioning and high-fidelity 3D content generation. It brings language, image, and 3D spatial content into a single framework, enabling the model to understand multimodal inputs and generate detailed 3D objects with strong spatial consistency.

Unlike prior 3D MLLM methods that often generate low-resolution meshes, textualized mesh tokens, or coarse structural proxies, CG-MLLM integrates a pretrained vision-language backbone with a specialized 3D VAE latent space. This design allows the model to perform end-to-end 3D generation within the MLLM paradigm while preserving fine-grained geometry.

## Highlights

- **Unified captioning and generation**: supports 3D understanding, captioning, and high-resolution 3D generation in one multimodal framework.
- **Language-image-3D integration**: aligns text tokens, visual tokens, and 3D spatial latent tokens in a shared modeling space.
- **Mixture-of-Transformer architecture**: decouples heterogeneous modeling requirements through TokenAR and BlockAR transformers.
- **Efficient high-resolution 3D modeling**: uses block-level autoregressive generation to reduce the cost of long 3D token sequences.
- **Strong 3D generation quality**: outperforms existing LLM-based 3D generation methods on multiple perceptual and semantic metrics reported in the paper.

## Method

CG-MLLM follows a decoder-only multimodal architecture with three major stages:

1. **Multimodal encoding**
   - Text is tokenized with the Qwen tokenizer.
   - Images are encoded with a Qwen3-VL-style vision encoder.
   - 3D assets are mapped into a high-order latent space through a frozen Spatial-VAE adapted from Hunyuan3D-2.1.

2. **Specialized MoT modeling**
   - **TokenAR Transformer** handles token-level autoregressive modeling for sequential content such as language.
   - **BlockAR Transformer** handles block-level parallel modeling for spatial 3D latent blocks.
   - A hybrid causal-parallel attention mask enables long-context interaction between standard tokens and spatial blocks.

3. **Multimodal decoding**
   - Text tokens are decoded into natural language.
   - Generated 3D latent tokens are decoded by the 3D VAE into geometric representations, followed by material synthesis for improved visual fidelity.

## Results

The paper reports that CG-MLLM achieves the best performance among compared MLLM-based 3D generation methods, including SAR3D and ShapeLLM-Omni.

| Model | p-FID ↓ | p-KID ↓ | CLIP-IQA+ ↑ | MUSIQ ↑ | Uni3D ↑ | CLIP ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAR3D | 30.07 | 1.00 | 0.42 | 66.01 | 0.3193 | 82.86 |
| ShapeLLM-Omni | 13.11 | 0.29 | 0.37 | 55.71 | 0.3190 | 84.18 |
| **CG-MLLM** | **12.55** | **0.27** | **0.45** | **71.65** | **0.3198** | **84.47** |

These results show that CG-MLLM improves both geometric fidelity and semantic alignment while keeping 3D generation inside a scalable MLLM framework.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{huang2026cgmllm,
  title={CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models},
  author={Huang, Junming and Wang, Chi and Li, Letian and Xu, Guangkai and Huang, Donglin and Chen, Hao and Dai, Qiang and Xu, Weiwei},
  journal={arXiv preprint arXiv:2601.21798},
  year={2026}
}
```

## Acknowledgements

We thank the open-source research community and the authors of the foundation models and 3D generation systems that make this research possible.
