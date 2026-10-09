# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import torch

from .data_utils import patchify_qwenvl
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform


class DataConfig:
    def __init__(
        self,
        grouped_datasets,
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.1,
        vit_patch_size=16,
        vit_merge_size=2,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.vit_merge_size = vit_merge_size


class PackedDataset(torch.utils.data.IterableDataset):
    """Samples from several dataset groups and packs them into one flex-attention sequence.

    Every sample is a `sequence_plan` of items:
      - text:        TokenAR tokens, CE loss when `loss` is 1
      - vit_image:   Qwen-VL ViT tokens (TokenAR)
      - vae_obj:     Hunyuan3D shape latents denoised by the BlockAR expert (flow-matching MSE)
      - und_vae_obj: PointBERT tokens of an object to be understood (BlockAR, no loss)
    """

    def __init__(
        self,
        data_config,
        tokenizer,
        special_tokens,
        local_rank,
        world_size,
        num_workers,
        expected_num_tokens=32768,
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.data_config = data_config
        for k, v in special_tokens.items():
            setattr(self, k, v)

        self.grouped_datasets, self.is_mandatory, self.grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.dataset_iters = [iter(dataset) for dataset in self.grouped_datasets]

    def build_datasets(self, datasets_metainfo, data_status):
        datasets, is_mandatory, grouped_weights = [], [], []
        for group_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))
            if 'image_transform_args' in dataset_args:
                dataset_args['transform'] = ImageTransform(**dataset_args.pop('image_transform_args'))

            dataset_args['data_dir_list'] = []
            for name in dataset_args.pop('dataset_names'):
                if self.local_rank == 0:
                    print(f'Preparing Dataset {group_name}/{name}')
                meta_info = DATASET_INFO[group_name][name]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])
                if 'jsonl_path' in meta_info:
                    dataset_args.setdefault('jsonl_path_list', []).append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and group_name in data_status and resume_data_status:
                group_data_status = data_status[group_name]
            else:
                group_data_status = None
            datasets.append(DATASET_REGISTRY[group_name](
                dataset_name=group_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=group_data_status,
                **dataset_args,
            ))
        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    @staticmethod
    def set_sequence_status():
        return dict(
            curr=0,
            sample_lens=[],
            split_lens=[],
            attn_modes=[],
            packed_position_ids=[[], [], []],  # mRoPE t / h / w
            packed_text_ids=[],
            packed_text_indexes=[],
            packed_label_ids=[],
            ce_loss_indexes=[],
            packed_vit_tokens=[],
            packed_vit_token_indexes=[],
            image_grid_thw=[],
            obj_surface=[],
            obj_vae_token_indexes=[],
            obj_vae_position_ids=[],
            obj_timesteps=[],
            obj_mse_loss_indexes=[],
            und_obj_surface=[],
            und_obj_vae_token_indexes=[],
        )

    def to_tensor(self, status):
        sequence_length = sum(status['sample_lens'])
        pad_len = self.max_num_tokens - sequence_length
        data = dict(
            sequence_length=sequence_length,
            sample_lens=status['sample_lens'] + [pad_len],
            split_lens=status['split_lens'] + [pad_len],
            attn_modes=status['attn_modes'] + ['causal'],
            packed_text_ids=torch.tensor(status['packed_text_ids']),
            packed_text_indexes=torch.tensor(status['packed_text_indexes']),
            packed_position_ids=torch.tensor(status['packed_position_ids']),
        )
        if len(status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(status['packed_vit_tokens'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(status['packed_vit_token_indexes'])
            data['image_grid_thw'] = torch.tensor(status['image_grid_thw'])
        if len(status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(status['ce_loss_indexes'])
        if len(status['obj_surface']) > 0:
            data['packed_obj_surface'] = torch.stack(status['obj_surface'], dim=0)
            data['packed_obj_vae_token_indexes'] = torch.tensor(status['obj_vae_token_indexes'])
            data['packed_obj_vae_position_ids'] = torch.cat(status['obj_vae_position_ids'], dim=0)
            data['obj_timesteps'] = torch.tensor(status['obj_timesteps'])
            data['obj_mse_loss_indexes'] = torch.tensor(status['obj_mse_loss_indexes'])
        if len(status['und_obj_surface']) > 0:
            data['packed_und_obj_surface'] = torch.stack(status['und_obj_surface'], dim=0)
            data['packed_und_obj_vae_token_indexes'] = torch.tensor(status['und_obj_vae_token_indexes'])
        return data

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []
        buffer = []
        while True:
            # Ensure at least one sample from each mandatory group.
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if not self.is_mandatory[group_index]:
                        continue
                    while True:
                        sample = next(group_iter)
                        # +2 per item for the start / end special tokens
                        num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                        if num_tokens < self.max_num_tokens_per_sample:
                            sequence_status = self.pack_sequence(sample, sequence_status)
                            batch_data_indexes.append(sample['data_indexes'])
                            break
                        print(f"skip a sample with length {num_tokens}, data_indexes: {sample['data_indexes']}")

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}, data_indexes: {sample['data_indexes']}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def _add_special_token(self, status, token_id, curr):
        status['packed_text_ids'].append(token_id)
        status['packed_text_indexes'].append(curr)

    def pack_sequence(self, sample, status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        surface_list = sample['surface_list']
        position_ids = status['packed_position_ids']

        split_lens, attn_modes = [], []
        curr = status['curr']
        curr_rope_id = 0  # RoPE restarts from 0 for every sample
        sample_len = 0

        for item in sample['sequence_plan']:
            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                status['packed_text_ids'].extend(shifted_text_ids)
                status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                self._add_special_token(status, self.eos_token_id, curr)
                curr += 1
                split_len = len(shifted_text_ids) + 1

                attn_modes.append('causal')
                for axis in position_ids:
                    axis.extend(range(curr_rope_id, curr_rope_id + split_len))
                curr_rope_id += split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                # keep at least one image per packed batch
                if (item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob
                        and len(status['image_grid_thw']) > 0):
                    curr_rope_id += 1
                    continue

                self._add_special_token(status, self.start_of_image, curr)
                curr += 1

                patch_size = self.data_config.vit_patch_size
                merge_size = self.data_config.vit_merge_size
                h, w = image_tensor.shape[1] // patch_size, image_tensor.shape[2] // patch_size
                vit_tokens = patchify_qwenvl(image_tensor, patch_size)
                num_img_tokens = vit_tokens.shape[0] // merge_size ** 2
                status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                status['packed_vit_tokens'].append(vit_tokens)
                status['image_grid_thw'].append((1, h, w))
                curr += num_img_tokens

                self._add_special_token(status, self.end_of_image, curr)
                curr += 1
                split_len = num_img_tokens + 2

                attn_modes.append('causal')
                grid_w = w // merge_size
                span = max(1, h // merge_size, grid_w)
                base = curr_rope_id + 1
                t_ids = [base] * num_img_tokens
                h_ids = [i // grid_w + base for i in range(num_img_tokens)]
                w_ids = [i % grid_w + base for i in range(num_img_tokens)]
                end_id = curr_rope_id + span + 1
                for axis, ids in zip(position_ids, (t_ids, h_ids, w_ids)):
                    axis.extend([curr_rope_id] + ids + [end_id])
                curr_rope_id += span + 2

            elif item['type'] == 'vae_obj':
                num_latent_tokens = item['num_tokens']
                self._add_special_token(status, self.start_of_obj, curr)
                curr += 1

                status['obj_surface'].append(surface_list.pop(0))
                status['obj_vae_position_ids'].append(torch.arange(num_latent_tokens))
                status['obj_vae_token_indexes'].extend(range(curr, curr + num_latent_tokens))
                if item['loss'] == 1:
                    status['obj_mse_loss_indexes'].extend(range(curr, curr + num_latent_tokens))
                    timestep = np.random.randn()
                else:
                    timestep = float('-inf')
                status['obj_timesteps'].extend([timestep] * num_latent_tokens)
                curr += num_latent_tokens

                self._add_special_token(status, self.end_of_obj, curr)
                curr += 1
                split_len = num_latent_tokens + 2

                attn_modes.append('noise' if item['loss'] == 1 else 'full')
                for axis in position_ids:
                    axis.extend([curr_rope_id] * split_len)
                if item['loss'] == 0:
                    curr_rope_id += 1

            elif item['type'] == 'und_vae_obj':
                num_latent_tokens = item['num_tokens']
                self._add_special_token(status, self.start_of_obj, curr)
                curr += 1

                status['und_obj_surface'].append(surface_list.pop(0))
                status['und_obj_vae_token_indexes'].extend(range(curr, curr + num_latent_tokens))
                curr += num_latent_tokens

                self._add_special_token(status, self.end_of_obj, curr)
                curr += 1
                split_len = num_latent_tokens + 2

                attn_modes.append('full')
                for axis in position_ids:
                    axis.extend([curr_rope_id] * split_len)
                curr_rope_id += 1

            else:
                raise ValueError(f"Unknown sequence_plan item type: {item['type']}")

            split_lens.append(split_len)
            sample_len += split_len

        status['curr'] = curr
        status['sample_lens'].append(sample_len)
        status['split_lens'].extend(split_lens)
        status['attn_modes'].extend(attn_modes)
        return status


class SimpleCustomBatch:
    def __init__(self, batch):
        self.data = batch[0]

    def _apply(self, fn):
        self.data = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.data.items()}
        return self

    def pin_memory(self):
        return self._apply(lambda t: t.pin_memory())

    def cuda(self, device):
        return self._apply(lambda t: t.to(device))

    def to_dict(self):
        return self.data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
