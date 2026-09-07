# Image VAE weights

Download `ae.safetensors` from [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) and place it in this folder:

```bash
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT ae.safetensors --local-dir models/BAGEL-7B-MoT
```

`vit_config.json` is already included for SigLIP-based checkpoints. Qwen-VL checkpoints do not need this file when `--use_qwen_vit` is set.
