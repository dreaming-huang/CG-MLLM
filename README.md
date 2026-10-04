<p align="center">
  <h1 align="center"><strong>CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models</strong></h1>
</p>

<p align="center">
    Junming Huang<sup>1,2</sup>,
    Chi Wang<sup>1</sup>,
    Letian Li<sup>1,2</sup>,
    Guangkai Xu<sup>1</sup>,
    Donglin Huang<sup>1</sup>,
    Hao Chen<sup>1</sup>,
    Qiang Dai<sup>2</sup>,
    Weiwei Xu<sup>1</sup>
    <br>
    <sup>1</sup>Zhejiang University,
    <sup>2</sup>LIGHTSPEED
</p>

<h3 align="center">ICML 2026</h3>

<div align="center">

<a href="https://arxiv.org/abs/2601.21798"><img src="https://img.shields.io/badge/arXiv-2601.21798-b31b1b.svg" alt="arXiv"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://cv.jream.top/CG-MLLM-page/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/JreamH/CGMLLM-4B-i2o"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-4B--i2o-orange" alt="4B weights"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/JreamH/CGMLLM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-v0.1-orange" alt="v0.1 weights"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/dreaming-huang/CG-MLLM"><img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="Code"></a>

</div>

<p align="center">
  <video src="assets/staircase_40_camera_1080p_hq.mp4" width="850" controls autoplay muted loop playsinline></video>
</p>

<p align="center">
  <img src="assets/overview.png" alt="CG-MLLM generation and understanding examples" width="850"/>
</p>

We present **CG-MLLM**, a unified multimodal large language model for 3D captioning and high-fidelity 3D content generation. CG-MLLM brings language, image, and 3D spatial content into a single framework, enabling multimodal understanding and detailed 3D object generation with strong spatial consistency.

This repository contains the **inference** code.

## News

- **2026-10-04**: **4B image-to-3D** checkpoint is released. It is a specialized training of our architecture for the image-to-3D object task on HY3D-Bench. Weights: [JreamH/CGMLLM-4B-i2o](https://huggingface.co/JreamH/CGMLLM-4B-i2o).
- **2026-09-07**: Inference code and **v0.1** checkpoint are released. Weights: [JreamH/CGMLLM](https://huggingface.co/JreamH/CGMLLM).
- **2026-05-07**: 🎉 Our paper **CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models** has been accepted to ICML 2026. See you in Seoul!



## Overview

Unlike prior 3D MLLM methods that often generate low-resolution meshes, textualized mesh tokens, or coarse structural proxies, CG-MLLM integrates a pretrained vision-language backbone with a specialized 3D VAE latent space. This design allows the model to perform end-to-end 3D generation within the MLLM paradigm while preserving fine-grained geometry.

<p align="center">
  <img src="assets/pipeline.png" alt="CG-MLLM pipeline" width="944"/>
</p>



## Installation

```bash
git clone https://github.com/dreaming-huang/CG-MLLM.git
cd CG-MLLM
conda create -n cgmllm python=3.10 -y
conda activate cgmllm
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

Download the BAGEL image VAE used to initialize the visual encoder config:

```bash
hf download ByteDance-Seed/BAGEL-7B-MoT ae.safetensors --local-dir models/BAGEL-7B-MoT
```

The Hunyuan3D-2.1 shape VAE is loaded from Hugging Face (`tencent/Hunyuan3D-2.1`) at runtime.

Download PointBERT. 
```bash
hf download RunsenXu/PointLLM_7B_v1.1_init point_bert_v1.1.pt --local-dir models
```

## Checkpoint

**4B image-to-3D** weights: **[JreamH/CGMLLM-4B-i2o](https://huggingface.co/JreamH/CGMLLM-4B-i2o)**.
This checkpoint is a specialized training of our architecture for the image-to-3D object task on HY3D-Bench, built on [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

```bash
hf download JreamH/CGMLLM-4B-i2o ema.safetensors --local-dir models/CGMLLM-4B-i2o
```

**v0.1** weights: **[JreamH/CGMLLM](https://huggingface.co/JreamH/CGMLLM)**.
This checkpoint is a multi-task training of our architecture, covering the various tasks described in the paper, built on [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).
Download the checkpoint so `--checkpoint` can find `ema.safetensors`:

```bash
hf download JreamH/CGMLLM ema.safetensors --local-dir models/CGMLLM
```



## Inference

4B image-to-3D object: 

```bash
python inference.py \
  --checkpoint models/CGMLLM-4B-i2o \
  --llm_base_path Qwen/Qwen3-VL-4B-Instruct \
  --use_qwen_vit --use_qwen_vl --qk_norm \
  --mode i2obj --bg white --border_ratio 0.15 \
  --image examples/chairo.png
```

v0.1 interactive menu:

```bash
python inference.py \
  --llm_base_path Qwen/Qwen3-VL-2B-Instruct \
  --checkpoint models/CGMLLM \
  --use_qwen_vit --use_qwen_vl --qk_norm \
  --timestep_shift 3.0 --img_cfg 7.5 --txt_cfg 7.5 \
  --bg transparent --border_ratio 0.45
```

v0.1 single-task examples:

```bash
# Image to 3D
python inference.py --checkpoint models/CGMLLM --use_qwen_vit --use_qwen_vl --qk_norm \
  --mode i2obj --bg transparent --border_ratio 0.45 --image examples/chairo.png

# Text to 3D
python inference.py --checkpoint models/CGMLLM --use_qwen_vit --use_qwen_vl --qk_norm \
  --mode t2obj --prompt "A medieval knight in T-pose, metallic armor with decorative trim, holding a sword and shield."

# Image understanding
python inference.py --checkpoint models/CGMLLM --use_qwen_vit --use_qwen_vl --qk_norm \
  --mode img_und --image examples/chairo.png --prompt "Describe the object in the image."

# 3D understanding with PointBERT
python inference.py --checkpoint models/CGMLLM --use_qwen_vit --use_qwen_vl --qk_norm \
  --mode obj_und --obj examples/0ea33b6617174530b97d6b7a92c275fb_8192.npy \
  --prompt "What does this collection of points represent?"
```

Generated meshes are written to `--output_dir` (default: `<checkpoint>/output_images`).

## Citation

If you find this work useful, please cite:

```bibtex
@article{huang2026cg,
  title={CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models},
  author={Huang, Junming and Wang, Chi and Li, Letian and Xu, Guangkai and Huang, Donglin and Chen, Hao and Dai, Qiang and Xu, Weiwei},
  journal={arXiv preprint arXiv:2601.21798},
  year={2026}
}
```

## Acknowledgements

This inference code is built on [BAGEL](https://github.com/ByteDance-Seed/Bagel) and the [Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) shape VAE, and uses the [Point-BERT](https://github.com/lulutang0608/Point-BERT) encoder released with [PointLLM](https://github.com/RunsenXu/PointLLM). Please follow their licenses when using those components.

We thank the open-source research community and the authors of the foundation models and 3D generation systems that make this research possible.