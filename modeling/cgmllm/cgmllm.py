# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import numpy as np
import os
from typing import List, Tuple, Optional
try:
    from peft import PeftModel
except ImportError:  # LoRA unwrap only; full-finetune checkpoints do not need peft
    PeftModel = ()
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
    patchify_qwenvl,
    packed_to_mrope_position_ids,
    packed_to_mrope_position_ids_repeat,
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding, PositionEmbedding1D,ZeroEmbedding
from contextlib import nullcontext
from tqdm import tqdm


class CGMLLMConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        obj_gen=True,
        obj_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        obj_vae_dim=64,
        und_obj_vae_dim=64,
        und_obj_vae_len=512,
        obj_vae_len=512,
        need_obj_pe=False,
        use_dinov2=False,
        dinov2_model_name="dinov2_vitl14_reg",
        dinov2_hidden_size=1024,
        dinov2_image_size=518,
        dinov2_patch_size=14,
        dinov2_fusion="gated_add",
        dinov2_gate_init=0.0,
        freeze_dinov2=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.obj_gen = obj_gen
        self.obj_und = obj_und

        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.obj_vae_dim = obj_vae_dim
        self.obj_vae_len = obj_vae_len
        self.need_obj_pe = need_obj_pe
        self.und_obj_vae_len = und_obj_vae_len
        self.und_obj_vae_dim = und_obj_vae_dim
        self.use_dinov2 = use_dinov2
        self.dinov2_model_name = dinov2_model_name
        self.dinov2_hidden_size = dinov2_hidden_size
        self.dinov2_image_size = dinov2_image_size
        self.dinov2_patch_size = dinov2_patch_size
        self.dinov2_fusion = dinov2_fusion
        self.dinov2_gate_init = dinov2_gate_init
        self.freeze_dinov2 = freeze_dinov2


class CGMLLM(PreTrainedModel):
    config_class = CGMLLMConfig
    base_model_prefix = 'cgmllm'

    def __init__(self, language_model, vit_model, config: CGMLLMConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        self._dinov2_fusion_logged = False

        if config.visual_gen or config.obj_gen:
            self.timestep_shift = config.timestep_shift
            self.time_embedder = TimestepEmbedder(self.hidden_size)


        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            # self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            # self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und :
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_hidden_size = config.vit_config.hidden_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            if self.config.vit_config.model_type =="siglip_vision_model":
                self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act) #two layer Linear
                self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)
            if config.use_dinov2:
                self.dinov2_connector = MLPconnector(config.dinov2_hidden_size, self.hidden_size, config.connector_act)
                self.dinov2_gate = nn.Parameter(torch.tensor([float(config.dinov2_gate_init)]))
                print(
                    "[CGMLLM-DINOv2] enabled "
                    f"model={config.dinov2_model_name} "
                    f"dino_hidden={config.dinov2_hidden_size} "
                    f"image_size={config.dinov2_image_size} "
                    f"patch_size={config.dinov2_patch_size} "
                    f"fusion={config.dinov2_fusion} "
                    f"gate_init={config.dinov2_gate_init} "
                    f"freeze_dinov2={config.freeze_dinov2}",
                    flush=True,
                )
                if config.freeze_dinov2:
                    object.__setattr__(self, "_dinov2_model", None)
                else:
                    self.dinov2_model = self._load_dinov2_model()

        if config.obj_gen:
            # self.timestep_shift = config.timestep_shift
            self.obj_vae_dim = config.obj_vae_dim
            self.obj2llm = nn.Linear(self.obj_vae_dim, self.hidden_size)
            self.llm2obj = nn.Linear(self.hidden_size, self.obj_vae_dim)
            self.gen_obj_pos_embed = PositionEmbedding1D(config.obj_vae_len, self.hidden_size) if config.need_obj_pe else ZeroEmbedding()

        if config.obj_und:
            self.und_obj_vae_dim = config.und_obj_vae_dim
            self.obj_und_connector = MLPconnector(self.und_obj_vae_dim, self.hidden_size, config.connector_act)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)
        if self.config.obj_gen:
            nn.init.constant_(self.llm2obj.weight, 0)
            nn.init.constant_(self.llm2obj.bias, 0)
        if getattr(self.config, "use_dinov2", False) and hasattr(self, "dinov2_gate"):
            nn.init.constant_(self.dinov2_gate, float(self.config.dinov2_gate_init))
    
    def unwrap_peft_model(self, m):
        if PeftModel and isinstance(m, PeftModel):
            return m.get_base_model()
        return m

    def _load_dinov2_model(self):
        print(f"[CGMLLM-DINOv2] loading torch.hub model {self.config.dinov2_model_name}", flush=True)
        model = torch.hub.load("facebookresearch/dinov2", self.config.dinov2_model_name, pretrained=True)
        model.eval()
        if self.config.freeze_dinov2:
            for param in model.parameters():
                param.requires_grad = False
        print("[CGMLLM-DINOv2] DINOv2 model loaded", flush=True)
        return model

    def _get_dinov2_model(self, device):
        if hasattr(self, "dinov2_model"):
            model = self.dinov2_model
        else:
            model = getattr(self, "_dinov2_model", None)
            if model is None:
                model = self._load_dinov2_model()
                object.__setattr__(self, "_dinov2_model", model)
        return model.to(device)

    def _prepare_dinov2_image(self, image):
        image = image.convert("RGB").resize(
            (self.config.dinov2_image_size, self.config.dinov2_image_size)
        )
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image - mean) / std

    def _encode_dinov2_images(self, packed_dino_images: torch.Tensor, device: torch.device) -> torch.Tensor:
        dinov2_model = self._get_dinov2_model(device)
        dinov2_device = next(dinov2_model.parameters()).device
        dinov2_dtype = next(dinov2_model.parameters()).dtype
        images = packed_dino_images.to(device=dinov2_device, dtype=dinov2_dtype)
        ctx = torch.no_grad() if self.config.freeze_dinov2 else nullcontext()
        with ctx:
            if hasattr(dinov2_model, "forward_features"):
                features = dinov2_model.forward_features(images)
            else:
                features = dinov2_model(images, is_training=True)
            if isinstance(features, dict):
                if "x_norm_patchtokens" in features:
                    patchtokens = features["x_norm_patchtokens"]
                elif "x_prenorm" in features:
                    patch_count = (images.shape[-2] // self.config.dinov2_patch_size) * (
                        images.shape[-1] // self.config.dinov2_patch_size
                    )
                    patchtokens = features["x_prenorm"][:, -patch_count:]
                else:
                    raise KeyError("DINOv2 output does not contain patch tokens.")
            else:
                patchtokens = features
            patchtokens = F.layer_norm(patchtokens, patchtokens.shape[-1:])
        return patchtokens.to(device)

    def _align_dinov2_tokens(
        self,
        dino_tokens: torch.Tensor,
        image_grid_thw: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if dino_tokens.ndim != 3:
            raise ValueError(f"Expected DINOv2 tokens with shape (B, N, D), got {tuple(dino_tokens.shape)}")
        dino_side = int(math.sqrt(dino_tokens.shape[1]))
        if dino_side * dino_side != dino_tokens.shape[1]:
            raise ValueError(f"DINOv2 patch token count must be square, got {dino_tokens.shape[1]}")

        spatial_merge_size = getattr(self.config.vit_config, "spatial_merge_size", 1)
        aligned_tokens = []
        for tokens, thw in zip(dino_tokens, image_grid_thw):
            target_h = int(thw[1].item()) // spatial_merge_size
            target_w = int(thw[2].item()) // spatial_merge_size
            token_map = tokens.transpose(0, 1).reshape(1, -1, dino_side, dino_side)
            token_map = F.interpolate(token_map.float(), size=(target_h, target_w), mode="bicubic", align_corners=False)
            aligned_tokens.append(token_map.reshape(token_map.shape[1], -1).transpose(0, 1))
        connector_device = self.dinov2_connector.fc1.weight.device
        connector_dtype = self.dinov2_connector.fc1.weight.dtype
        aligned_tokens = torch.cat(aligned_tokens, dim=0).to(device=connector_device, dtype=connector_dtype)
        return self.dinov2_connector(aligned_tokens).to(dtype=target_dtype)

    def _fuse_dinov2_tokens(
        self,
        packed_vit_token_embed: torch.Tensor,
        packed_dino_images: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not getattr(self.config, "use_dinov2", False) or packed_dino_images is None or image_grid_thw is None:
            return packed_vit_token_embed
        dino_tokens = self._encode_dinov2_images(packed_dino_images, packed_vit_token_embed.device)
        dino_embed = self._align_dinov2_tokens(dino_tokens, image_grid_thw, packed_vit_token_embed.dtype)
        dino_embed = dino_embed.to(device=packed_vit_token_embed.device, dtype=packed_vit_token_embed.dtype)
        if dino_embed.shape[0] != packed_vit_token_embed.shape[0]:
            raise ValueError(
                f"DINOv2/Qwen token count mismatch: {dino_embed.shape[0]} vs {packed_vit_token_embed.shape[0]}"
            )
        if not self._dinov2_fusion_logged:
            rank = os.environ.get("RANK", "0")
            gate_value = self.dinov2_gate.detach().float().item() if hasattr(self, "dinov2_gate") else None
            print(
                "[CGMLLM-DINOv2] first fusion "
                f"rank={rank} "
                f"qwen_tokens={tuple(packed_vit_token_embed.shape)} "
                f"dino_tokens={tuple(dino_tokens.shape)} "
                f"dino_projected={tuple(dino_embed.shape)} "
                f"image_grid_thw={image_grid_thw.detach().cpu().tolist()} "
                f"fusion={self.config.dinov2_fusion} "
                f"gate={gate_value} "
                f"requires_grad_connector={any(p.requires_grad for p in self.dinov2_connector.parameters())} "
                f"requires_grad_gate={self.dinov2_gate.requires_grad}",
                flush=True,
            )
            self._dinov2_fusion_logged = True
        if self.config.dinov2_fusion == "add":
            return packed_vit_token_embed + dino_embed
        if self.config.dinov2_fusion == "gated_add":
            return packed_vit_token_embed + self.dinov2_gate.to(packed_vit_token_embed.dtype) * dino_embed
        raise ValueError(f"Unsupported DINOv2 fusion mode: {self.config.dinov2_fusion}")
    
    def run_vit_microbatch(
        self,
        packed_vit_tokens,
        image_grid_thw,
        expect_tokens=4096,
    ):
        outputs = []
        start = 0
        img_ptr = 0

        while start < packed_vit_tokens.shape[0] and img_ptr < len(image_grid_thw):
            curr_tokens = 0
            batch_tokens = []
            batch_thw = []
            # Run once curr_tokens reaches expect_tokens; also flush the last batch.
            while img_ptr < len(image_grid_thw):
                thw = image_grid_thw[img_ptr]
                num = int(thw[0] * thw[1] * thw[2])

                batch_tokens.append(packed_vit_tokens[start:start + num])
                batch_thw.append(thw)
                curr_tokens += num
                start += num
                img_ptr += 1

                if curr_tokens >= expect_tokens:
                    break

            batch_tokens = torch.cat(batch_tokens, dim=0)
            batch_thw = torch.stack(batch_thw, dim=0)

            out = self.vit_model(batch_tokens, batch_thw)
            outputs.append(out)

        return torch.cat(outputs, dim=0)
    


    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor, 
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_embed: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,  # token index in the sentence
        packed_vit_position_ids: Optional[torch.LongTensor] = None,  # token index inside one image
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        image_grid_thw:Optional[torch.Tensor] = None,
        packed_dino_images: Optional[torch.Tensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        #for object generation
        packed_obj_vae_tokens: Optional[torch.Tensor] = None,
        packed_obj_vae_position_ids: Optional[torch.LongTensor] = None,
        packed_obj_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_obj_vae_shapes: Optional[List[int]] = None,
        obj_mse_loss_indexes: Optional[torch.BoolTensor] = None,
        obj_timesteps: Optional[torch.LongTensor] = None,

        #for object understanding
        packed_und_obj_vae_tokens: Optional[torch.Tensor] = None,
        packed_und_obj_vae_position_ids: Optional[torch.LongTensor] = None,
        packed_und_obj_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_und_obj_vae_shapes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.unwrap_peft_model(self.language_model).model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size)) #l,h fill with zeros
        packed_sequence[packed_text_indexes] = packed_text_embedding #packed_text_indexes records the position of packed_text_embedding in packed_sequence

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        
        if "vl" in self.language_model.config.model_type and len(packed_position_ids.shape)==1:
            with torch.no_grad():
                packed_position_ids=packed_position_ids.unsqueeze(0)  # originally (seq_len,)
                spatial_merge_size = self.config.vit_config.spatial_merge_size if hasattr(self.config.vit_config, "spatial_merge_size") else 1
                packed_position_ids,_=packed_to_mrope_position_ids(packed_position_ids,packed_vit_token_indexes,image_grid_thw,spatial_merge_size=spatial_merge_size)
        elif "vl" in self.language_model.config.model_type and len(packed_position_ids.shape)==2:
            packed_position_ids=packed_position_ids.unsqueeze(1)  # originally (3, seq_len)

        deepstack_image_embeds=None
        if self.config.visual_und and self.config.vit_config.model_type =="siglip_vision_model":
            if vit_token_seqlens is not None:
                cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)) #
                cu_seqlens = cu_seqlens.to(torch.int32)
                max_seqlen = torch.max(vit_token_seqlens).item()
                packed_vit_token_embed = self.vit_model( #calculate vit token embeddings(like a CLIP) , packed token will be seperated by flash attn using cu_seqlens
                    packed_pixel_values=packed_vit_tokens, 
                    packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                packed_vit_token_embed = self.connector(packed_vit_token_embed)
                vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
        else:
            if packed_vit_token_embed is not None:
                pass
            elif image_grid_thw is not None:
                ctx = torch.no_grad() if self.config.llm_config.freeze_vit else nullcontext()
                with ctx:
                    out=self.vit_model(packed_vit_tokens,image_grid_thw)
                    if not isinstance(out, tuple):
                        packed_vit_token_embed=out
                    else:
                        packed_vit_token_embed=out[0]
                        deepstack_image_embeds=out[1]


                    # packed_vit_token_embed = []
                    # deepstack_image_embeds = []
                    # start_idx = 0
                    # if image_grid_thw is not None:
                    #     for thw in image_grid_thw:  # token length of each image
                    #         length = int(thw[0] * thw[1] * thw[2])
                    #         end_idx = start_idx + length
                    #         # current image tokens
                    #         tokens_chunk = packed_vit_tokens[start_idx:end_idx]
                    #         out = self.vit_model(tokens_chunk, thw.unsqueeze(0))
                    #         if not isinstance(out, tuple):
                    #             out = (out,)
                    #         packed_vit_token_embed.append(out[0])
                    #         if len(out) > 1: deepstack_image_embeds.append(out[1])
                    #         start_idx = end_idx
                    #     packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)
                    #     packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

                    #     layers = list(zip(*deepstack_image_embeds))
                    #     layers = [torch.cat(layer, dim=0) for layer in layers]
                    #     deepstack_image_embeds = torch.stack(layers, dim=0) if len(layers) else None

        if packed_vit_token_embed is not None and packed_vit_token_indexes is not None:
            packed_vit_token_embed = self._fuse_dinov2_tokens(
                packed_vit_token_embed,
                packed_dino_images,
                image_grid_thw,
            )
            if packed_vit_token_embed.dtype != packed_sequence.dtype:
                packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        # Allocate tensors for timesteps.
        packed_all_timestep_embeds = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes): #h,w is the num of patch(pre define)
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)

            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent



        if self.config.obj_gen and packed_obj_vae_tokens is not None:
            packed_obj_vae_clean = packed_obj_vae_tokens
            obj_noise = torch.randn_like(packed_obj_vae_clean)
            obj_timesteps = torch.sigmoid(obj_timesteps)
            obj_timesteps = self.timestep_shift * obj_timesteps / (1 + (self.timestep_shift - 1) * obj_timesteps)
            packed_obj_vae = (1 - obj_timesteps[:, None]) * packed_obj_vae_clean + obj_timesteps[:, None] * obj_noise

            packed_obj_timestep_embeds = self.time_embedder(obj_timesteps)
            obj_token_pos_emb = self.gen_obj_pos_embed(packed_obj_vae_position_ids)

            packed_obj_vae = self.obj2llm(packed_obj_vae) + packed_obj_timestep_embeds + obj_token_pos_emb

            packed_sequence[packed_obj_vae_token_indexes] = packed_obj_vae
            packed_all_timestep_embeds[packed_obj_vae_token_indexes] = packed_obj_timestep_embeds
        
        if self.config.obj_und and packed_und_obj_vae_tokens is not None:
            packed_und_obj_vae_tokens = self.obj_und_connector(packed_und_obj_vae_tokens)
            packed_und_obj_vae = packed_und_obj_vae_tokens
            packed_sequence[packed_und_obj_vae_token_indexes] = packed_und_obj_vae



        extra_inputs = {}
        if self.use_moe: 

            # Initialize packed lists.
            network_token_ar_indexes_list = [packed_text_indexes, packed_vit_token_indexes]
            network_block_ar_indexes_list = [
                packed_vae_token_indexes,
                packed_obj_vae_token_indexes,
                packed_und_obj_vae_token_indexes,
            ]

            network_token_ar_indexes = torch.cat([x for x in network_token_ar_indexes_list if x is not None], dim=0)
            network_block_ar_indexes = torch.cat([x for x in network_block_ar_indexes_list if x is not None], dim=0)
            extra_inputs.update(
                packed_token_ar_indexes=network_token_ar_indexes,
                packed_block_ar_indexes=network_block_ar_indexes,
            )
        # prepare visual pos masks
        visual_pos_masks=None
        if packed_vit_token_indexes is not None:
            visual_pos_masks = torch.zeros(packed_sequence.size(0), dtype=torch.bool, device=packed_sequence.device)
            visual_pos_masks[packed_vit_token_indexes] = True

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds,
            packed_all_timestep_embeds=packed_all_timestep_embeds,
            **extra_inputs,
        )

        mse = 0
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse += ((packed_mse_preds - target[has_mse]) ** 2).mean(dim=-1, keepdim=True)
        if self.config.obj_gen and packed_obj_vae_tokens is not None:
            packed_obj_mse_preds = self.llm2obj(last_hidden_state[obj_mse_loss_indexes])
            target = obj_noise - packed_obj_vae_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_obj_mse = obj_timesteps > 0
            obj_mse = ((packed_obj_mse_preds - target[has_obj_mse]) ** 2).mean(dim=-1, keepdim=True)
            # mse = torch.cat([mse, obj_mse], dim=0)
            if isinstance(mse, torch.Tensor):
                mse = torch.cat([mse, obj_mse], dim=0)
            else:
                mse = obj_mse

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        if not isinstance(mse, torch.Tensor):
            mse = None

        return dict(mse=mse, ce=ce)

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            if not "vl" in self.language_model.config.model_type:
                packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids))) # to be specified [0, 1, 2, 3, 4, 5, 6, 7] for text
            else:
                for _ in range(3):  # 3 axes
                    packed_text_position_ids[_][0].extend(range(curr_position_id, curr_position_id + len(text_ids)))# to be specified [0, 1, 2, 3, 4] [0, 1, 2, 3, 4] [0, 1, 2, 3, 4]for text


            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        # Append the assistant start token at the end.
        # assistant_str = "\n<|im_start|>assistant\n"
        # assistant_ids = tokenizer.encode(assistant_str)

        # packed_text_ids.extend(assistant_ids)
        # text_token_lens.append(len(assistant_ids))

        # if not "vl" in self.language_model.config.model_type:
        #     packed_text_position_ids.extend(
        #         range(new_rope[-1], new_rope[-1] + len(assistant_ids))
        #     )
        # else:
        #     for d in range(3):
        #         packed_text_position_ids[d][0].extend(
        #             range(new_rope[-1], new_rope[-1] + len(assistant_ids))
        #         )

        # packed_text_indexes.extend(
        #     range(curr, curr + len(assistant_ids))
        # )
        # curr += len(assistant_ids)
        # newlens[-1] += len(assistant_ids)
        # new_rope[-1] += len(assistant_ids)
        
        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.unwrap_peft_model(self.language_model).model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "token_ar"}

        
        if "vl" in self.language_model.config.model_type and len(packed_text_position_ids.shape)==1:
            with torch.no_grad():
                packed_text_position_ids=packed_text_position_ids.unsqueeze(0)  # originally (seq_len,)
                packed_text_position_ids,_=packed_to_mrope_position_ids_repeat(packed_text_position_ids)
                packed_text_position_ids=packed_text_position_ids.to(packed_text_embedding.device)
        elif "vl" in self.language_model.config.model_type and len(packed_text_position_ids.shape)==2:
            packed_text_position_ids=packed_text_position_ids.unsqueeze(1)  # originally (3, seq_len)
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids.to(packed_text_embedding.device),
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_dino_images = list()
        image_grid_thw=list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_indexes = list(), list()
        packed_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[], [], []])
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen  # curr is global, _curr is local

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            if getattr(self.config, "use_dinov2", False):
                packed_dino_images.append(self._prepare_dinov2_image(image))
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            # vit_tokens = patchify(image_tensor, self.vit_patch_size)
            _, _h, _w = image_tensor.shape
            h,w=_h//self.vit_patch_size,_w//self.vit_patch_size

            if self.config.vit_config.model_type =="siglip_vision_model":
                vit_tokens = patchify(image_tensor, self.vit_patch_size)
            else:
                vit_tokens = patchify_qwenvl(image_tensor, self.vit_patch_size) #temporal_patch_size =2,merge_size =2


            packed_vit_tokens.append(vit_tokens)
            if self.config.vit_config.model_type =="siglip_vision_model":
                num_img_tokens = vit_tokens.shape[0]
            else:
                num_img_tokens = vit_tokens.shape[0]//(2**2)
            
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            image_grid_thw.append((1, h, w))
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            if not "vl" in self.language_model.config.model_type:
                packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2)) # to be specified [8, 8, 8, 8, 8, 8, 8, 8] for image
                new_rope.append(curr_position_id + 1)

            else:
                h_len=h//self.config.vit_config.spatial_merge_size
                w_len=w//self.config.vit_config.spatial_merge_size
                t, h, w = 1,h_len,w_len
                # Expand 3D coords in t->h->w order.
                t_ids = [i // (h * w)+curr_position_id+1 for i in range(num_img_tokens)]  # [0,0,0,...,0]
                h_ids = [(i % (h * w)) // w +curr_position_id+1 for i in range(num_img_tokens)]  # [0,0,...,1,1,...,h-1,h-1,...]
                w_ids = [i % w +curr_position_id+1 for i in range(num_img_tokens)]  # [0,1,2,...,w-1, 0,1,2,...,w-1, ...]
                packed_position_ids[0].extend([curr_position_id]+t_ids+[curr_position_id+max(t,h,w)+1])
                packed_position_ids[1].extend([curr_position_id]+h_ids+[curr_position_id+max(t,h,w)+1])
                packed_position_ids[2].extend([curr_position_id]+w_ids+[curr_position_id+max(t,h,w)+1])
                new_rope.append(curr_position_id + 2+max(t,h,w))

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "image_grid_thw": torch.tensor(image_grid_thw, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0), #Order for vit tokens in image
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long), #Disorder for language tokens in conversation
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        if len(packed_dino_images) > 0:
            generation_input["packed_dino_images"] = torch.stack(packed_dino_images, dim=0)

        return generation_input, newlens, new_rope

    
    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_vit_token_embed: torch.Tensor = None,
        image_grid_thw: torch.IntTensor = None,
        packed_dino_images: torch.Tensor = None,
    ):
        packed_text_embedding = self.unwrap_peft_model(self.language_model).model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        # cu_seqlens = cu_seqlens.to(torch.int32)
        # max_seqlen = torch.max(vit_token_seqlens).item()

        # packed_vit_token_embed = self.vit_model(
        #     packed_pixel_values=packed_vit_tokens, 
        #     packed_flattened_position_ids=packed_vit_position_ids,
        #     cu_seqlens=cu_seqlens,
        #     max_seqlen=max_seqlen,
        # )
        # packed_vit_token_embed = self.connector(packed_vit_token_embed)
        # pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        # packed_vit_token_embed = packed_vit_token_embed + pos_emb
        deepstack_image_embeds=None
        if self.config.vit_config.model_type =="siglip_vision_model":
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)) #
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model( #calculate vit token embeddings(like a CLIP) , packed token will be seperated by flash attn using cu_seqlens
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
        else:
            if packed_vit_token_embed is not None:
                packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
            else:
                ctx = torch.no_grad()
                with ctx:
                    packed_vit_token_embed = []
                    deepstack_image_embeds = []
                    start_idx = 0
                    for thw in image_grid_thw:  # token length of each image
                        length = int(thw[0] * thw[1] * thw[2])
                        end_idx = start_idx + length
                        # current image tokens
                        tokens_chunk = packed_vit_tokens[start_idx:end_idx]
                        out = self.vit_model(tokens_chunk, thw.unsqueeze(0))
                        if not isinstance(out, tuple):
                            out = (out,)
                        packed_vit_token_embed.append(out[0])
                        if len(out) > 1: deepstack_image_embeds.append(out[1])
                        start_idx = end_idx
                    packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)

                    layers = list(zip(*deepstack_image_embeds))
                    layers = [torch.cat(layer, dim=0) for layer in layers]
                    deepstack_image_embeds = torch.stack(layers, dim=0) if len(layers) else None


        packed_vit_token_embed = self._fuse_dinov2_tokens(
            packed_vit_token_embed,
            packed_dino_images,
            image_grid_thw,
        )
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed


        # prepare visual pos masks
        visual_pos_masks=None
        if packed_vit_token_indexes is not None:
            visual_pos_masks = torch.zeros(packed_sequence.size(0), dtype=torch.bool, device=packed_sequence.device)
            visual_pos_masks[packed_vit_token_indexes] = True


        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "token_ar"}
        
        if "vl" in self.language_model.config.model_type and len(packed_position_ids.shape)==1:
            with torch.no_grad():
                packed_position_ids=packed_position_ids.unsqueeze(0)  # originally (seq_len,)
                spatial_merge_size = self.config.vit_config.spatial_merge_size if hasattr(self.config.vit_config, "spatial_merge_size") else 1
                packed_position_ids,_=packed_to_mrope_position_ids(packed_position_ids,packed_vit_token_indexes,image_grid_thw,spatial_merge_size=spatial_merge_size)
                packed_position_ids=packed_position_ids.to(packed_sequence.device)
        elif "vl" in self.language_model.config.model_type and len(packed_position_ids.shape)==2:
            packed_position_ids=packed_position_ids.unsqueeze(1)  # originally (3, seq_len)
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids.to(packed_sequence.device),
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds,
            is_causal=False if self.config.vit_config.model_type =="siglip_vision_model" or not "vl" in self.language_model.config.model_type else True,  # causal only when both ViT and LLM come from Qwen-VL
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_indexes = list(), list()
        packed_key_value_indexes = list()
        packed_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            if not "vl" in self.language_model.config.model_type:
                packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            else:
                for _ in range(3):  # 3 axes
                    packed_position_ids[_][0].extend([curr_position_id] * (num_img_tokens + 2))


            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.unwrap_peft_model(self.language_model).model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "block_ar",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        if "vl" in self.language_model.config.model_type and len(packed_position_ids.shape)==1:
            with torch.no_grad():
                packed_position_ids=packed_position_ids.unsqueeze(0)  # originally (seq_len,)
                packed_position_ids,_=packed_to_mrope_position_ids_repeat(packed_position_ids)
                packed_position_ids=packed_position_ids.to(packed_sequence.device)
        elif "vl" in self.language_model.config.model_type and  len(packed_position_ids.shape)==2:
            packed_position_ids=packed_position_ids.unsqueeze(1)  # originally (3, seq_len)

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids.to(packed_sequence.device),
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values


    def prepare_und_obj(self, curr_kvlens, curr_rope, obj_latents, new_token_ids, timestep=0):
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_indexes = list(), list()
        packed_key_value_indexes = list()
        packed_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])

        _curr = curr = 0
        vae_obj_tensors = list()
        newlens, new_rope = list(), list()
        for obj_latent, curr_kvlen, curr_position_id in zip(obj_latents, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_obj'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            vae_obj_tensors.append(obj_latent)

            num_img_tokens = obj_latent.shape[1]
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_obj'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            if not "vl" in self.language_model.config.model_type:
                packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            else:
                for _ in range(3):  # 3 axes
                    packed_position_ids[_][0].extend([curr_position_id] * (num_img_tokens + 2))


            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)


        padded_objs = torch.cat(vae_obj_tensors, dim=0)

        generation_input = {
            "padded_objs": padded_objs,
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_obj(  # used by understanding; generation does not cache objects
        self,
        past_key_values: NaiveCache,
        padded_objs: torch.Tensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.unwrap_peft_model(self.language_model).model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        packed_latent = padded_objs
        packed_timestep_embeds = self.time_embedder(packed_timesteps)

        packed_all_timestep_embeds = packed_text_embedding.new_zeros(size=(sum(packed_seqlens), self.hidden_size))

        packed_latent = self.obj_und_connector(packed_latent)

        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)

        packed_sequence[packed_vae_token_indexes] = packed_latent
        packed_all_timestep_embeds[packed_vae_token_indexes] = packed_timestep_embeds

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "block_ar",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        if "vl" in self.language_model.config.model_type and len(packed_position_ids.shape)==1:
            with torch.no_grad():
                packed_position_ids=packed_position_ids.unsqueeze(0)  # originally (seq_len,)
                packed_position_ids,_=packed_to_mrope_position_ids_repeat(packed_position_ids)
                packed_position_ids=packed_position_ids.to(packed_sequence.device)
        elif "vl" in self.language_model.config.model_type and  len(packed_position_ids.shape)==2:
            packed_position_ids=packed_position_ids.unsqueeze(1)  # originally (3, seq_len)

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids.to(packed_sequence.device),
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            packed_all_timestep_embeds=packed_all_timestep_embeds,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()

        packed_seqlens, packed_indexes = list(), list()
        packed_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])

        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            if not "vl" in self.language_model.config.model_type:
                packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            else:
                for _ in range(3):  # 3 axes
                    packed_position_ids[_][0].extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_indexes, packed_key_value_indexes =list(), list()
        packed_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            if not "vl" in self.language_model.config.model_type:
                packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            else:
                for _ in range(3):  # 3 axes
                    packed_position_ids[_][0].extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input
    
    def prepare_vae_obj_latent(self, curr_kvlens, curr_rope, obj_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_obj_vae_position_ids, packed_obj_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_seqlens, packed_indexes = list(), list()
        packed_key_value_indexes = list()
        packed_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])



        query_curr = curr = 0
        for num_latent_tokens, curr_kvlen, curr_position_id in zip(obj_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_obj'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            obj_vae_posiiton_ids = torch.arange(num_latent_tokens)
            packed_obj_vae_position_ids.append(obj_vae_posiiton_ids)
            packed_init_noises.append(
                torch.randn(num_latent_tokens, self.obj_vae_dim)
            )
            packed_obj_vae_token_indexes.extend(range(query_curr, query_curr + num_latent_tokens))
            packed_indexes.extend(range(curr, curr + num_latent_tokens))
            curr += num_latent_tokens
            query_curr += num_latent_tokens

            packed_text_ids.append(new_token_ids['end_of_obj'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            if not "vl" in self.language_model.config.model_type:
                packed_position_ids.extend([curr_position_id] * (num_latent_tokens + 2))
            else:
                for _ in range(3):  # 3 axes
                    packed_position_ids[_][0].extend([curr_position_id] * (num_latent_tokens + 2))


            packed_seqlens.append(num_latent_tokens + 2)
        
        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            # "packed_obj_vae_position_ids": torch.cat(packed_obj_vae_position_ids, dim=0),
            # "packed_obj_vae_token_indexes": torch.tensor(packed_obj_vae_token_indexes, dtype=torch.long),
            "packed_vae_position_ids": torch.cat(packed_obj_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_obj_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input
    
    def prepare_vae_obj_latent_cfg(self, curr_kvlens, curr_rope, obj_sizes):
        packed_indexes, packed_key_value_indexes = list(), list()
        packed_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])


        query_curr = curr = 0
        for num_latent_tokens, curr_kvlen, curr_position_id in zip(obj_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_indexes.extend(range(curr, curr + num_latent_tokens))
            curr += num_latent_tokens
            query_curr += num_latent_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1
            if not "vl" in self.language_model.config.model_type:
                packed_position_ids.extend([curr_position_id] * (num_latent_tokens + 2))
            else:
                for _ in range(3):  # 3 axes
                    packed_position_ids[_][0].extend([curr_position_id] * (num_latent_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input



    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent
    
    @torch.no_grad
    def generate_obj(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                gen_obj=True,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        gen_obj: bool = False,
    ): 
        packed_text_embedding = self.unwrap_peft_model(self.language_model).model.embed_tokens(packed_text_ids) #ids are tokens
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size)) #embed tokens
        packed_sequence[packed_text_indexes] = packed_text_embedding #indexes are the indexes of the tokens in the packed sequence
        packed_all_timestep_embeds = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size)) #embed tokens

        assert timestep.unique().shape[0] == 1
        packed_timestep_embeds = self.time_embedder(timestep)
        if gen_obj:  # switch between object and image generation
            packed_pos_embed = self.gen_obj_pos_embed(packed_vae_position_ids)
            x_t = self.obj2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        else:
            packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
            x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed

        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)

        packed_sequence[packed_vae_token_indexes] = x_t
        packed_all_timestep_embeds[packed_vae_token_indexes] = packed_timestep_embeds

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "block_ar",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids.to(packed_sequence.device),
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes, #condition token indexes
            update_past_key_values=False,
            is_causal=False,
            packed_all_timestep_embeds=packed_all_timestep_embeds,
            **extra_inputs,
        )
        # v_t = self.llm2vae(output.packed_query_sequence)
        if gen_obj:
            v_t = self.llm2obj(output.packed_query_sequence)
        else:
            v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference( #results without text condition
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids.to(packed_sequence.device),
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                packed_all_timestep_embeds=packed_all_timestep_embeds,
                **extra_inputs,
            )
            # cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            if gen_obj:
                cfg_text_v_t = self.llm2obj(cfg_text_output.packed_query_sequence) 
            else:
                cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids.to(packed_sequence.device),
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                packed_all_timestep_embeds=packed_all_timestep_embeds,
                **extra_inputs,
            )
            # cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            if gen_obj:
                cfg_img_v_t = self.llm2obj(cfg_img_output.packed_query_sequence)
            else:
                cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale #rescale to v_t's norm
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids= list() if not "vl" in self.language_model.config.model_type else list([[[]], [[]], [[]]])


        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            if not "vl" in self.language_model.config.model_type:
                packed_query_position_ids.append(curr_position_id)
            else:
                for _ in range(3):  # 3 axes
                    packed_query_position_ids[_][0].append(curr_position_id)
            curr += curr_kvlen
        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.unwrap_peft_model(self.language_model).model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "token_ar"}
            # print("packed_query_position_ids.shape:",packed_query_position_ids)
            if "vl" in self.language_model.config.model_type and len(packed_query_position_ids.shape)==1:
                with torch.no_grad():
                    packed_query_position_ids=packed_query_position_ids.unsqueeze(0)  # originally (seq_len,)
                    packed_query_position_ids,_=packed_to_mrope_position_ids_repeat(packed_query_position_ids)
                    packed_query_position_ids=packed_query_position_ids.to(packed_text_embedding.device)
            elif "vl" in self.language_model.config.model_type and len(packed_query_position_ids.shape)==2:
                packed_query_position_ids=packed_query_position_ids.unsqueeze(1)  # originally (3, seq_len)


            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids.to(packed_text_embedding.device),
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)
        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output
