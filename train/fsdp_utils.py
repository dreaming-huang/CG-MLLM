# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from modeling.cgmllm.modeling_utils import MLPconnector
from modeling.cgmllm.qwen2_vl_navit import Qwen2VLDecoderLayer, Qwen2VLMoTDecoderLayer

WRAPPED_MODULES = (Qwen2VLDecoderLayer, Qwen2VLMoTDecoderLayer, MLPconnector)


class FSDPConfig:
    def __init__(self, sharding_strategy, backward_prefetch, cpu_offload, num_replicate, num_shard=8):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=set(WRAPPED_MODULES)),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=True,
    )


def grad_checkpoint_check_fn(module):
    return isinstance(module, WRAPPED_MODULES)


def fsdp_ema_setup(ema_model, fsdp_config):
    for param in ema_model.parameters():
        param.requires_grad = False
    return fsdp_wrapper(ema_model, fsdp_config)


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params, new_params = [], []
    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)


def clone_shared_tensors(state_dict):
    """safetensors refuses tensors that share storage (e.g. tied embeddings); clone the duplicates."""
    seen = set()
    cloned = {}
    for name, tensor in state_dict.items():
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr(), storage.nbytes(), tensor.storage_offset(),
               tuple(tensor.size()), tuple(tensor.stride()))
        cloned[name] = tensor.detach().cpu().clone() if key in seen else tensor
        seen.add(key)
    return cloned


def _optimizer_shard(fsdp_config):
    if fsdp_config.sharding_strategy == "FULL_SHARD":
        return dist.get_rank(), dist.get_world_size()
    if fsdp_config.sharding_strategy == "HYBRID_SHARD":
        return dist.get_rank() % fsdp_config.num_shard, fsdp_config.num_shard
    raise NotImplementedError(fsdp_config.sharding_strategy)


def _load_model_weights(path):
    state_dict = load_file(path, device="cpu")
    # Sinusoidal position tables are rebuilt from the config, so sizes may differ across stages.
    return {k: v for k, v in state_dict.items() if not k.endswith("pos_embed.pos_embed")}


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(ckpt_dir, train_steps, model, ema_model, optimizer, scheduler, data_status, logger, fsdp_config):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        full_state_cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        for name, module in (("ema", ema_model), ("model", model)):
            with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, full_state_cfg):
                state_dict = module.state_dict()
                if dist.get_rank() == 0:
                    save_file(clone_shared_tensors(state_dict), os.path.join(save_path, f"{name}.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            shard_index, total_shards = _optimizer_shard(fsdp_config)
            if dist.get_rank() == shard_index:
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"),
                )

        if dist.get_rank() == 0:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            if data_status is not None:
                torch.save(data_status, os.path.join(save_path, "data_status.pt"))
        dist.barrier()

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model, resume_from_ema=False):
        if resume_from is None or not os.path.exists(resume_from):
            logger.info("Training from scratch.")
            return model, ema_model

        logger.info(f"Loading checkpoint from {resume_from}.")
        model_path = os.path.join(resume_from, "model.safetensors")
        ema_path = os.path.join(resume_from, "ema.safetensors")
        if resume_from_ema or not os.path.exists(model_path):
            model_path = ema_path
        if not os.path.exists(ema_path):
            ema_path = model_path

        logger.info(model.load_state_dict(_load_model_weights(model_path), strict=False))
        logger.info(ema_model.load_state_dict(_load_model_weights(ema_path), strict=False))
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        """Returns (optimizer, scheduler, train_steps, data_status of this rank)."""
        if resume_from is None or not os.path.exists(resume_from):
            return optimizer, scheduler, 0, None

        shard_index, total_shards = _optimizer_shard(fsdp_config)
        optimizer_path = os.path.join(resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt")
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
        except Exception as e:
            print(f"Failed to load optimizer state from {optimizer_path}: {e}")
        try:
            scheduler.load_state_dict(
                torch.load(os.path.join(resume_from, "scheduler.pt"), map_location="cpu", weights_only=True)
            )
        except Exception as e:
            print(f"Failed to load scheduler state: {e}")

        train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1

        # data_status: [ {dataset_name: {worker_id: data_indexes}} for each rank ]
        data_status = None
        data_status_path = os.path.join(resume_from, "data_status.pt")
        if os.path.exists(data_status_path):
            all_status = torch.load(data_status_path, map_location="cpu", weights_only=True)
            if dist.get_rank() < len(all_status):
                data_status = all_status[dist.get_rank()]
        return optimizer, scheduler, train_steps, data_status
