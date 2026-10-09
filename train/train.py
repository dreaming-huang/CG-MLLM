# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import datetime
import functools
import gc
import os
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from time import time
from typing import Optional

import torch
import torch.distributed as dist
import wandb
import yaml
from easydict import EasyDict
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from data.data_utils import add_special_tokens  # noqa: E402
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper  # noqa: E402
from data.dataset_info import OBJ_GEN_GROUPS  # noqa: E402
from hy3dshape.models.autoencoders import ShapeVAE  # noqa: E402
from modeling.cgmllm import CGMLLM, CGMLLMConfig, Qwen2VLConfig, Qwen2VLForConditionalGeneration  # noqa: E402
from modeling.qwen2 import Qwen2Tokenizer  # noqa: E402
from pointbert.point_encoder import PointTransformer  # noqa: E402
from train.fsdp_utils import (  # noqa: E402
    FSDPCheckpoint,
    FSDPConfig,
    fsdp_ema_setup,
    fsdp_ema_update,
    fsdp_wrapper,
    grad_checkpoint_check_fn,
)
from train.train_utils import create_logger, get_latest_ckpt  # noqa: E402

POINTBERT_CONFIG = os.path.join(REPO_ROOT, "pointbert", "PointTransformer_base_8192point.yaml")
POINTBERT_DIM = 1152
POINTBERT_TOKENS = 513  # 512 point groups + cls


@dataclass
class ModelArguments:
    llm_path: str = field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        metadata={"help": "Qwen-VL checkpoint (Qwen2-VL / Qwen2.5-VL / Qwen3-VL); provides the LLM, ViT and tokenizer."},
    )
    layer_module: str = field(
        default="Qwen2VLMoTDecoderLayer",
        metadata={"help": "Qwen2VLMoTDecoderLayer (TokenAR + BlockAR experts) or Qwen2VLDecoderLayer (shared weights)."},
    )
    llm_qk_norm: bool = field(default=True, metadata={"help": "QK RMSNorm inside attention."})
    tie_word_embeddings: bool = field(default=True, metadata={"help": "Tie input and output embeddings."})
    obj_vae_path: str = field(
        default="tencent/Hunyuan3D-2.1",
        metadata={"help": "Hunyuan3D-2.1 shape VAE that encodes surfaces into BlockAR latents."},
    )
    obj_vae_len: int = field(
        default=512,
        metadata={"help": "Number of shape latents per object (512 / 1024 / 2048 / 4096)."},
    )
    und_obj_vae_path: str = field(
        default="models/point_bert_v1.1.pt",
        metadata={"help": "PointBERT checkpoint used to encode point clouds for 3D understanding."},
    )
    need_obj_pe: bool = field(default=False, metadata={"help": "Add 1D position embeddings to shape latents."})
    text_cond_dropout_prob: float = field(default=0.1, metadata={"help": "Drop rate of text conditions (for CFG)."})
    vit_cond_dropout_prob: float = field(default=0.1, metadata={"help": "Drop rate of image conditions (for CFG)."})


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/cgmllm_multitask.yaml",
        metadata={"help": "YAML file with dataset groups, weights and preprocessing."},
    )
    prefetch_factor: int = field(default=2, metadata={"help": "Batches pre-loaded by each DataLoader worker."})
    num_workers: int = field(default=4, metadata={"help": "DataLoader workers."})
    max_num_tokens_per_sample: int = field(default=8192, metadata={"help": "Longer samples are skipped."})
    max_num_tokens: int = field(default=20480, metadata={"help": "Hard limit of tokens in a packed batch."})
    prefer_buffer_before: int = field(
        default=8192,
        metadata={"help": "While the batch is shorter than this, pop from the overflow buffer first."},
    )
    max_buffer_size: int = field(default=50, metadata={"help": "Max oversized samples kept in the overflow buffer."})
    data_seed: int = field(default=42, metadata={"help": "Seed for shuffling data shards."})


@dataclass
class TrainingArguments:
    results_dir: str = field(default="results", metadata={"help": "Directory for logs."})
    checkpoint_dir: str = field(default="results/checkpoints", metadata={"help": "Directory for checkpoints."})
    wandb_project: str = field(default="cgmllm", metadata={"help": "W&B project."})
    wandb_name: str = field(default="run", metadata={"help": "W&B run name."})
    wandb_mode: str = field(default="online", metadata={"help": "W&B mode: online, offline or disabled."})

    global_seed: int = field(default=4396, metadata={"help": "Base seed, offset by rank."})
    auto_resume: bool = field(default=False, metadata={"help": "Resume from the latest checkpoint in checkpoint_dir."})
    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint directory to start from, used when auto_resume finds nothing."},
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only weights from resume_from (e.g. the previous stage), not optimizer / step / data."},
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "With resume_model_only, initialize the model from ema.safetensors."},
    )

    log_every: int = field(default=10, metadata={"help": "Log every N steps."})
    save_every: int = field(default=2000, metadata={"help": "Save a checkpoint every N steps."})
    total_steps: int = field(default=500_000, metadata={"help": "Total optimizer steps."})
    warmup_steps: int = field(default=2000, metadata={"help": "Linear warm-up steps."})
    lr_scheduler: str = field(default="constant", metadata={"help": "constant or cosine."})
    lr: float = field(default=1e-4, metadata={"help": "Peak learning rate."})
    min_lr: float = field(default=1e-7, metadata={"help": "Final learning rate of the cosine schedule."})
    beta1: float = field(default=0.9, metadata={"help": "AdamW beta1."})
    beta2: float = field(default=0.95, metadata={"help": "AdamW beta2."})
    eps: float = field(default=1e-15, metadata={"help": "AdamW epsilon."})
    ema: float = field(default=0.9999, metadata={"help": "EMA decay."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Gradient clipping norm."})
    timestep_shift: float = field(default=1.0, metadata={"help": "Shift of flow-matching timesteps."})
    mse_weight: float = field(default=1.0, metadata={"help": "Weight of the BlockAR flow-matching loss."})
    ce_weight: float = field(default=1.0, metadata={"help": "Weight of the TokenAR cross-entropy loss."})
    expected_num_tokens: int = field(default=20000, metadata={"help": "Yield a packed batch once it reaches this size."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Micro-batches per optimizer step."})

    num_replicate: int = field(default=1, metadata={"help": "HYBRID_SHARD replicas (usually the number of nodes)."})
    num_shard: int = field(default=8, metadata={"help": "HYBRID_SHARD shards (usually GPUs per node)."})
    sharding_strategy: str = field(default="HYBRID_SHARD", metadata={"help": "FULL_SHARD or HYBRID_SHARD."})
    backward_prefetch: str = field(default="BACKWARD_PRE", metadata={"help": "BACKWARD_PRE or NO_PREFETCH."})
    cpu_offload: bool = field(default=False, metadata={"help": "Offload FSDP parameters to CPU."})
    grad_ckpt: bool = field(default=True, metadata={"help": "Activation checkpointing on decoder layers."})

    freeze_llm: bool = field(default=False, metadata={"help": "Freeze the whole language model."})
    freeze_vit: bool = field(default=False, metadata={"help": "Freeze the Qwen-VL ViT."})
    freeze_token_ar: bool = field(
        default=False,
        metadata={"help": "Freeze the TokenAR expert and train only the BlockAR expert."},
    )
    init_block_ar_from_token_ar: bool = field(
        default=True,
        metadata={"help": "Initialize the BlockAR expert as a copy of the pretrained TokenAR weights."},
    )
    obj_und: bool = field(
        default=True,
        metadata={"help": "Train 3D understanding (PointBERT tokens)."},
    )


def load_qwen_vl(model_args, training_args, logger):
    """Build the packed MoT language model from a Qwen-VL checkpoint and return (llm, llm_config, vit)."""
    llm_config = Qwen2VLConfig.from_pretrained(model_args.llm_path)
    if getattr(llm_config, "text_config", None) is not None:
        for key, value in llm_config.text_config.items():
            setattr(llm_config, key, value)
        del llm_config.text_config
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_token_ar = training_args.freeze_token_ar
    llm_config.freeze_vit = training_args.freeze_vit

    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_args.llm_path, attn_implementation="flash_attention_2"
    )
    language_model = Qwen2VLForConditionalGeneration(llm_config)
    target_keys = language_model.state_dict().keys()
    state_dict = {
        k.replace("model.language_model.", "model."): v
        for k, v in hf_model.state_dict().items()
        if not k.startswith("model.visual.")
    }
    msg = language_model.load_state_dict({k: v for k, v in state_dict.items() if k in target_keys}, strict=False)
    missing = [k for k in msg.missing_keys if "_block_ar" not in k]
    logger.info(f"LLM weights loaded from {model_args.llm_path}, missing (excluding BlockAR): {missing}")
    if training_args.init_block_ar_from_token_ar:
        language_model.init_moe()

    vit_model = hf_model.visual
    del hf_model
    return language_model, llm_config, vit_model


def load_shape_vae(path, num_latents):
    kwargs = dict(use_safetensors=False)
    if num_latents != 4096:
        kwargs.update(num_latents=num_latents, pc_size=num_latents * 20, pc_sharpedge_size=0)
    return ShapeVAE.from_pretrained(path, **kwargs)


def load_pointbert(ckpt_path):
    with open(POINTBERT_CONFIG, "r") as f:
        config = EasyDict(yaml.safe_load(f)).model
    config.point_dims = 6  # xyz + rgb
    model = PointTransformer(config, use_max_pool=False)
    model.load_checkpoint(ckpt_path)
    return model


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.wandb_name,
            resume="allow",
            mode=training_args.wandb_mode,
            config={**asdict(model_args), **asdict(data_args), **asdict(training_args)},
        )
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")
    logger.info(f"Training arguments {training_args}")

    resume_from = training_args.resume_from
    resume_model_only = training_args.resume_model_only
    if training_args.auto_resume:
        latest_ckpt = get_latest_ckpt(training_args.checkpoint_dir)
        if latest_ckpt is not None:
            resume_from, resume_model_only = latest_ckpt, False
    finetune_from_ema = resume_model_only and training_args.finetune_from_ema

    set_seed(training_args.global_seed * dist.get_world_size() + dist.get_rank())

    # Model: Qwen-VL backbone with TokenAR (text / ViT) and BlockAR (3D latents / point clouds) experts.
    language_model, llm_config, vit_model = load_qwen_vl(model_args, training_args, logger)
    vit_config = vit_model.config
    config = CGMLLMConfig(
        visual_gen=False,
        visual_und=True,
        obj_gen=True,
        obj_und=training_args.obj_und,
        llm_config=llm_config,
        vit_config=vit_config,
        timestep_shift=training_args.timestep_shift,
        obj_vae_len=model_args.obj_vae_len,
        need_obj_pe=model_args.need_obj_pe,
        und_obj_vae_dim=POINTBERT_DIM,
        und_obj_vae_len=POINTBERT_TOKENS,
    )
    model = CGMLLM(language_model, vit_model, config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # Frozen external encoders.
    obj_vae = load_shape_vae(model_args.obj_vae_path, model_args.obj_vae_len)
    encoders = [obj_vae]
    und_obj_encoder = None
    if training_args.obj_und:
        und_obj_encoder = load_pointbert(model_args.und_obj_vae_path)
        encoders.append(und_obj_encoder)
    for encoder in encoders:
        encoder.requires_grad_(False)
        encoder.to(device).eval()

    if training_args.freeze_llm:
        model.language_model.requires_grad_(False)
    if training_args.freeze_token_ar:
        for name, param in model.language_model.named_parameters():
            if "_block_ar" not in name:
                param.requires_grad = False
    if training_args.freeze_vit:
        model.vit_model.eval()
        model.vit_model.requires_grad_(False)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params / 1e9:.2f}B, trainable: {num_trainable / 1e9:.2f}B")

    # FSDP, EMA and checkpoint loading.
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    if training_args.grad_ckpt:
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=grad_checkpoint_check_fn,
        )

    optimizer = torch.optim.AdamW(
        [p for p in fsdp_model.parameters() if p.requires_grad],
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
        fused=True,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps)
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")

    if resume_model_only:
        train_step, data_status = 0, None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config
        )

    # Packed dataloader.
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)
    if not training_args.obj_und and "pointllm_pretrain" in dataset_meta:
        raise ValueError("pointllm_pretrain needs 3D understanding; remove it from the config or set --obj_und True.")
    for group_name, group_cfg in dataset_meta.items():
        if "image_transform_args" in group_cfg:
            group_cfg["image_transform_args"]["image_stride"] = vit_config.patch_size
            group_cfg["image_transform_args"]["merge_size"] = vit_config.spatial_merge_size
        if group_name in OBJ_GEN_GROUPS:
            group_cfg["latent_length"] = model_args.obj_vae_len
    dataset_config = DataConfig(
        grouped_datasets=dataset_meta,
        text_cond_dropout_prob=model_args.text_cond_dropout_prob,
        vit_cond_dropout_prob=model_args.vit_cond_dropout_prob,
        vit_patch_size=vit_config.patch_size,
        vit_merge_size=vit_config.spatial_merge_size,
    )
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # one packed sequence per step
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor if data_args.num_workers > 0 else None,
    )

    last_saved_step = None

    def save_checkpoint(step):
        nonlocal last_saved_step
        last_saved_step = step
        torch.cuda.empty_cache()
        gather_list = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(data_status, gather_list, dst=0)
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir,
            train_steps=step,
            model=fsdp_model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list,
        )
        gc.collect()
        torch.cuda.empty_cache()

    fsdp_model.train()
    ema_model.eval()
    world_size = dist.get_world_size()
    accum_steps = training_args.gradient_accumulation_steps
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    curr_step = train_step
    start_time = time()
    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // accum_steps
        if curr_step >= training_args.total_steps:
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop("batch_data_indexes")
        # Skip the gradient all-reduce on accumulation micro-steps.
        is_accum_step = (micro_step + 1) % accum_steps != 0

        with fsdp_model.no_sync() if is_accum_step else nullcontext():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    if "packed_obj_surface" in data:
                        latents = obj_vae.encode(data.pop("packed_obj_surface"))
                        data["packed_obj_vae_tokens"] = latents.reshape(-1, latents.shape[-1])
                    if "packed_und_obj_surface" in data:
                        point_feats = und_obj_encoder(data.pop("packed_und_obj_surface")[..., :6].contiguous())
                        data["packed_und_obj_vae_tokens"] = point_feats.reshape(-1, point_feats.shape[-1])
                loss_dict = fsdp_model(**data)

                # Normalize by the number of loss tokens over all ranks.
                num_loss_tokens = torch.tensor(
                    [len(data.get("ce_loss_indexes", [])), len(data.get("obj_mse_loss_indexes", []))],
                    device=device, dtype=torch.float,
                )
                dist.all_reduce(num_loss_tokens)
                total_ce_tokens, total_mse_tokens = num_loss_tokens.tolist()
                loss = 0
                ce = torch.tensor(0.0, device=device)
                mse = torch.tensor(0.0, device=device)
                if loss_dict["ce"] is not None:
                    ce = loss_dict["ce"].sum() * world_size / total_ce_tokens
                    loss = loss + ce * training_args.ce_weight
                if loss_dict["mse"] is not None:
                    mse = loss_dict["mse"].mean(dim=-1).sum() * world_size / total_mse_tokens
                    loss = loss + mse * training_args.mse_weight
                loss = loss / accum_steps
            loss.backward()

        if not is_accum_step:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()

            if curr_step % training_args.log_every == 0:
                losses = torch.stack([ce.detach(), mse.detach()]).float()
                dist.all_reduce(losses)
                ce_value, mse_value = (losses / world_size).tolist()
                steps_per_sec = training_args.log_every / max(time() - start_time, 1e-6)
                mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024 ** 2, device=device)
                dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
                logger.info(
                    f"(step={curr_step:07d}) Train Loss ce: {ce_value:.4f}, Train Loss mse: {mse_value:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if dist.get_rank() == 0:
                    wandb.log({
                        "ce": ce_value,
                        "mse": mse_value,
                        "lr": optimizer.param_groups[0]["lr"],
                        "total_norm": total_norm.item(),
                        "total_ce_tokens": total_ce_tokens * accum_steps,
                        "total_mse_tokens": total_mse_tokens * accum_steps,
                        "steps_per_sec": steps_per_sec,
                        "mem_allocated": mem_allocated.item(),
                    }, step=curr_step)
                start_time = time()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            data_status.setdefault(item["dataset_name"], {})[item["worker_id"]] = item["data_indexes"]

        if not is_accum_step and curr_step > 0 and curr_step % training_args.save_every == 0:
            save_checkpoint(curr_step)

    if curr_step > 0 and curr_step != last_saved_step:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        save_checkpoint(curr_step)

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
