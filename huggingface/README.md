---
license: apache-2.0
library_name: transformers
pipeline_tag: image-to-3d
base_model: Qwen/Qwen3-VL-2B-Instruct
tags:
  - 3d
  - image-to-3d
  - text-to-3d
  - multimodal
  - qwen3-vl
  - mllm
language:
  - en
paper: arxiv:2601.21798
---

# CG-MLLM v0.1

**CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models** (ICML 2026)

Junming Huang, Chi Wang, Letian Li, Guangkai Xu, Donglin Huang, Hao Chen, Qiang Dai, Weiwei Xu

This repo hosts the **v0.1** inference weights (`ema.safetensors`). Use them with the [official inference code](https://github.com/dreaming-huang/CG-MLLM).

- Paper: [arXiv:2601.21798](https://arxiv.org/abs/2601.21798)
- Project page: [cv.jream.top/CG-MLLM-page](https://cv.jream.top/CG-MLLM-page/)
- Code: [github.com/dreaming-huang/CG-MLLM](https://github.com/dreaming-huang/CG-MLLM)

## Model description

CG-MLLM is a unified multimodal LLM for 3D captioning and high-fidelity 3D generation. It combines a pretrained [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) backbone with a [Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) Spatial-VAE, using a Mixture-of-Transformers design:

- **TokenAR** for token-level (text / image) modeling
- **BlockAR** for parallel 3D latent-block generation (4096 tokens)

The released file is an EMA checkpoint for inference. The Hunyuan3D-2.1 shape VAE is loaded separately at runtime and is not included here.

## Usage

```bash
hf download JreamH/CGMLLM ema.safetensors --local-dir models/CGMLLM

git clone https://github.com/dreaming-huang/CG-MLLM.git
cd CG-MLLM
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation

# Default 3D understanding encoder: models/point_bert_v1.1.pt
hf download RunsenXu/PointLLM_7B_v1.1_init point_bert_v1.1.pt --local-dir models

python inference.py \
  --llm_base_path Qwen/Qwen3-VL-2B-Instruct \
  --checkpoint models/CGMLLM \
  --use_qwen_vit --use_qwen_vl --qk_norm \
  --timestep_shift 3.0 --img_cfg 7.5 --txt_cfg 7.5 \
  --mode i2obj --bg transparent --border_ratio 0.45 \
  --image examples/chairo.png
```

`--checkpoint` must be a directory that contains `ema.safetensors`.

## Citation

```bibtex
@misc{huang2026cgmllmcaptioninggenerating3d,
      title={CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models},
      author={Junming Huang and Chi Wang and Letian Li and Guangkai Xu and Donglin Huang and Hao Chen and Qiang Dai and Weiwei Xu},
      year={2026},
      eprint={2601.21798},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.21798},
}
```

## License

Apache-2.0 for this checkpoint and the inference code. Please also follow the licenses of [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [BAGEL](https://github.com/ByteDance-Seed/Bagel), and [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) when using those components.
