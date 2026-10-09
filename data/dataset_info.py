# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .hy3d_dataset import HY3DFullIterableDataset
from .i2obj_dataset import I2ObjIterableDataset
from .pointllm_dataset import PointLLMJSONLIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'pointllm_pretrain': PointLLMJSONLIterableDataset,
    'i2obj_pretrain': I2ObjIterableDataset,
    't2obj_pretrain': I2ObjIterableDataset,  # same loader, use cond_mode: text_only in the yaml
    'hy3d_i2obj_pretrain': HY3DFullIterableDataset,
}

# Groups that generate shape latents; their latent_length is set from --obj_vae_len.
OBJ_GEN_GROUPS = ('i2obj_pretrain', 't2obj_pretrain', 'hy3d_i2obj_pretrain')

# Point these paths at your local copies of the data.
DATASET_INFO = {
    'vlm_sft': {
        'llava_recap_558k': {
            'data_dir': 'datasets/LLaVA-ReCap-558K/images',
            'jsonl_path': 'datasets/LLaVA-ReCap-558K/metadata.jsonl',
        },
        'llava_onevision': {
            'data_dir': 'datasets/LLaVA-OneVision-Data/images',
            'jsonl_path': 'datasets/LLaVA-OneVision-Data/metadata.jsonl',
        },
    },
    'pointllm_pretrain': {
        'pointllm_brief': {
            'data_dir': 'datasets/PointLLM/8192_npy',
            'jsonl_path': 'datasets/PointLLM/PointLLM_brief_description_660K_filtered.jsonl',
        },
    },
    'i2obj_pretrain': {
        'trellis_sketchfab': {'data_dir': 'datasets/CGMLLM_t500k_sketchfab'},
        'trellis_github': {'data_dir': 'datasets/CGMLLM_t500k_github'},
    },
    'hy3d_i2obj_pretrain': {
        'hy3d_full_train': {'data_dir': 'datasets/HY3D-Bench/full/train'},
    },
}
DATASET_INFO['t2obj_pretrain'] = DATASET_INFO['i2obj_pretrain']
