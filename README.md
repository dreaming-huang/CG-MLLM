# CG-MLLM

**[ICML 2026] CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models**

Junming Huang, Chi Wang, Letian Li, Guangkai Xu, Donglin Huang, Hao Chen, Qiang Dai, Weiwei Xu

[[Paper](https://arxiv.org/abs/2601.21798)]

## News

- **2026-05-07**: 🎉 Our paper **CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models** has been accepted to ICML 2026. See you in Seoul!

## Overview

**CG-MLLM** is a unified multimodal large language model for 3D captioning and high-fidelity 3D content generation. It brings language, image, and 3D spatial content into a single framework, enabling the model to understand multimodal inputs and generate detailed 3D objects with strong spatial consistency.

Unlike prior 3D MLLM methods that often generate low-resolution meshes, textualized mesh tokens, or coarse structural proxies, CG-MLLM integrates a pretrained vision-language backbone with a specialized 3D VAE latent space. This design allows the model to perform end-to-end 3D generation within the MLLM paradigm while preserving fine-grained geometry.

## Results

| Method | p-FID(↓) | p-KID(↓) | clipiqa+(↑) | musiq(↑) | clip(↑) | user-study(↑) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Diffusion-Base** |  |  |  |  |  |  |
| michelangelo | 17.96 | 0.56 | 0.45 | 71.42 | 84.08 | 2.6 |
| craftsman | 14.09 | 0.4 | 0.45 | 71.09 | 84.86 | 3.15 |
| hunyuan3d-2.1 | 16.8 | 0.53 | 0.47 | 71.2 | 85.11 | 3.15 |
| trellis | 7.36 | 0.12 | 0.44 | 66.97 | 84.13 | 3.28 |
| sam3d | 33.92 | 1.13 | 0.47 | 70.21 | 84.67 | 3.45 |
| **MLLM-Base** |  |  |  |  |  |  |
| sar3d | 30.07 | 1 | 0.42 | 66.01 | 82.86 | 2.93 |
| shapellm-omni | 13.11 | 0.29 | 0.37 | 55.71 | 84.18 | 2.3 |
| **ours** | **12.55** | **0.27** | **0.45** | **71.65** | **84.47** | **3.32** |

## Acknowledgements

We thank the open-source research community and the authors of the foundation models and 3D generation systems that make this research possible.
