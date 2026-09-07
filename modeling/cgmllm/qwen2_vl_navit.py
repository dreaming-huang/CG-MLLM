# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput

from flash_attn import flash_attn_varlen_func
from modeling.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLAttention, 
    Qwen2MLP, 
    Qwen2VLPreTrainedModel, 
    Qwen2RMSNorm,
    Qwen2VLRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from modeling.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig as _Qwen2VLConfig

try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

    class Qwen3VLTextRotaryEmbedding(nn.Module):
        inv_freq: torch.Tensor  # fix linting for `register_buffer`

        def __init__(self, config, device=None):
            super().__init__()
            if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", "default")
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

            self.config = config
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

            self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

        def apply_interleaved_mrope(self, freqs, mrope_section):
            """Apply interleaved MRoPE to 3D rotary embeddings.
            Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
            interleaved [THTHWHTHW...TT], preserving frequency continuity.
            args:
                x: (3, bs, seq_len, head_dim // 2)
                mrope_section: (3,)
            returns:
                x_t: (bs, seq_len, head_dim // 2)
            """
            freqs_t = freqs[0]  # just overwrite the first dimension T
            for dim, offset in enumerate((1, 2), start=1):  # H, W
                length = mrope_section[dim] * 3
                idx = slice(offset, length, 3)
                freqs_t[..., idx] = freqs[dim, ..., idx]
            return freqs_t

        @torch.no_grad()
        @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
        def forward(self, x, position_ids):
            # In contrast to other models, Qwen3VL has different position ids for the grids
            # So we expand the inv_freq to shape (3, ...)
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):  # Force float32
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
                freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
except ImportError:
    print("Qwen3VLTextRotaryEmbedding import failed.")


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
flex_attention = torch.compile(flex_attention)



class Qwen2VLConfig(_Qwen2VLConfig):
    def __init__(
        self,
        *args,
        qk_norm=True,
        layer_module="Qwen2VLDecoderLayer",
        freeze_token_ar=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_token_ar = freeze_token_ar

class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0



@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class PackedAttentionVL(Qwen2VLAttention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        # QKV projection
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        # Norm
        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        # multimodal RoPE
        packed_cos, packed_sin = packed_position_embeddings
        if self.config.architectures[0] == "Qwen3VLForConditionalGeneration":
            packed_query_states, packed_key_states = apply_rotary_pos_emb(
                packed_query_states.unsqueeze(0).transpose(1,2),
                packed_key_states.unsqueeze(0).transpose(1,2),
                packed_cos.unsqueeze(0),
                packed_sin.unsqueeze(0),
            )
        else:
            packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb(
                packed_query_states.unsqueeze(0).transpose(1,2),
                packed_key_states.unsqueeze(0).transpose(1,2),
                packed_cos,
                packed_sin,
                self.config.rope_scaling["mrope_section"],
            )
        packed_query_states=packed_query_states.squeeze(0).transpose(1,2)


        if isinstance(attention_mask, List):
            packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            packed_key_states = pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states.unsqueeze(0), 
                packed_key_states.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)
        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        # QKV Projection
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        # multimodal RoPE
        packed_cos, packed_sin = packed_query_position_embeddings

        if self.config.architectures[0] == "Qwen3VLForConditionalGeneration":
            packed_query_states, packed_key_states = apply_rotary_pos_emb(
                packed_query_states.unsqueeze(0).transpose(1,2),
                packed_key_states.unsqueeze(0).transpose(1,2),
                packed_cos.unsqueeze(0),
                packed_sin.unsqueeze(0),
            )
        else:
            packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb(
                packed_query_states.unsqueeze(0).transpose(1,2),
                packed_key_states.unsqueeze(0).transpose(1,2),
                packed_cos,
                packed_sin,
                self.config.rope_scaling["mrope_section"],
            )
        packed_query_states=packed_query_states.squeeze(0).transpose(1,2)

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]
            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_value_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class PackedAttentionMoTVL(Qwen2VLAttention):
    """MoT attention with a TokenAR expert (default projections) and a BlockAR expert (`*_block_ar`)."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_block_ar = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_block_ar = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_block_ar = nn.Identity()
            self.k_norm_block_ar = nn.Identity()

        self.q_proj_block_ar = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=getattr(config,"attention_bias",True))
        self.k_proj_block_ar = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config,"attention_bias",True))
        self.v_proj_block_ar = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config,"attention_bias",True))
        self.o_proj_block_ar = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)
    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_token_ar_indexes: torch.LongTensor,
        packed_block_ar_indexes: torch.LongTensor,
    ):
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))

        packed_sequence_token_ar = packed_sequence[packed_token_ar_indexes]
        packed_sequence_block_ar = packed_sequence[packed_block_ar_indexes]

        packed_query_states[packed_token_ar_indexes] = self.q_proj(packed_sequence_token_ar)
        packed_query_states[packed_block_ar_indexes] = self.q_proj_block_ar(packed_sequence_block_ar)

        packed_key_states[packed_token_ar_indexes] = self.k_proj(packed_sequence_token_ar)
        packed_key_states[packed_block_ar_indexes] = self.k_proj_block_ar(packed_sequence_block_ar)

        packed_value_states[packed_token_ar_indexes] = self.v_proj(packed_sequence_token_ar)
        packed_value_states[packed_block_ar_indexes] = self.v_proj_block_ar(packed_sequence_block_ar)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        if self.config.freeze_token_ar:
            packed_value_states[packed_token_ar_indexes] = packed_value_states[packed_token_ar_indexes].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_token_ar_indexes] = self.q_norm(packed_query_states[packed_token_ar_indexes])
        if self.config.freeze_token_ar:
            packed_query_states_[packed_token_ar_indexes] = packed_query_states_[packed_token_ar_indexes].detach()
        packed_query_states_[packed_block_ar_indexes] = self.q_norm_block_ar(packed_query_states[packed_block_ar_indexes])

        packed_key_states_[packed_token_ar_indexes] = self.k_norm(packed_key_states[packed_token_ar_indexes])
        if self.config.freeze_token_ar:
            packed_key_states_[packed_token_ar_indexes] = packed_key_states_[packed_token_ar_indexes].detach()
        packed_key_states_[packed_block_ar_indexes] = self.k_norm_block_ar(packed_key_states[packed_block_ar_indexes])

        # multimodal RoPE
        packed_cos, packed_sin = packed_position_embeddings
        if self.config.architectures[0] == "Qwen3VLForConditionalGeneration":
            packed_query_states_, packed_key_states_  = apply_rotary_pos_emb(
                packed_query_states_.unsqueeze(0).transpose(1,2),
                packed_key_states_.unsqueeze(0).transpose(1,2),
                packed_cos.unsqueeze(0) if len(packed_cos.shape)==2 else packed_cos,
                packed_sin.unsqueeze(0) if len(packed_sin.shape)==2 else packed_sin,
                )
        else:
            packed_query_states_, packed_key_states_ = apply_multimodal_rotary_pos_emb(
                packed_query_states_.unsqueeze(0).transpose(1,2),
                packed_key_states_.unsqueeze(0).transpose(1,2),
                packed_cos,
                packed_sin,
                self.config.rope_scaling["mrope_section"],
            )
        pad_size = sum(sample_lens) - packed_query_states.shape[0]
        
        packed_query_states_=packed_query_states_.squeeze(0).transpose(0,1) #len,head,dim
        packed_key_states_=packed_key_states_.squeeze(0).transpose(0,1) #len,head,dim


        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size).contiguous()
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size).contiguous()
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size).contiguous()
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape[0], self.hidden_size)
        packed_attn_output_[packed_token_ar_indexes] = self.o_proj(packed_attn_output[packed_token_ar_indexes])
        if self.config.freeze_token_ar:
            packed_attn_output_[packed_token_ar_indexes] = packed_attn_output_[packed_token_ar_indexes].detach()
        packed_attn_output_[packed_block_ar_indexes] = self.o_proj_block_ar(packed_attn_output[packed_block_ar_indexes])

        return packed_attn_output_

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="token_ar",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == 'token_ar':
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == 'block_ar':
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
            packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_block_ar(packed_vae_query_sequence)

            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_block_ar(packed_vae_query_sequence)

            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_block_ar(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_block_ar(packed_query_states[packed_vae_token_indexes])

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_block_ar(packed_key_states[packed_vae_token_indexes])

        # multimodal RoPE
        packed_cos, packed_sin = packed_query_position_embeddings #torch.Size([3, 1, 434, 128])
        if self.config.architectures[0] == "Qwen3VLForConditionalGeneration":
            packed_query_states, packed_key_states  = apply_rotary_pos_emb(
                packed_query_states.unsqueeze(0).transpose(1,2),
                packed_key_states.unsqueeze(0).transpose(1,2),
                packed_cos.unsqueeze(0) if len(packed_cos.shape)==2 else packed_cos,
                packed_sin.unsqueeze(0) if len(packed_sin.shape)==2 else packed_sin,
                )
        else:
            packed_query_states, packed_key_states = apply_multimodal_rotary_pos_emb( #torch.Size([1, head, len, dim]) torch.Size([1, head, len, dim])
                packed_query_states.unsqueeze(0).transpose(1,2),
                packed_key_states.unsqueeze(0).transpose(1,2),
                packed_cos,
                packed_sin,
                self.config.rope_scaling["mrope_section"],
            )
        
        packed_query_states=packed_query_states.squeeze(0).transpose(0,1)
        packed_key_states=packed_key_states.squeeze(0).transpose(0,1)

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))
        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        # packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = packed_attn_output.reshape(-1, self.num_heads * self.head_dim)

        if mode == 'token_ar':
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'block_ar':
            packed_attn_output_ = packed_attn_output.new_zeros(
                packed_attn_output.shape[0],
                self.hidden_size,
            )
            packed_attn_output_[packed_text_indexes] = self.o_proj(
                packed_attn_output[packed_text_indexes]
            )
            packed_attn_output_[packed_vae_token_indexes] = self.o_proj_block_ar(
                packed_attn_output[packed_vae_token_indexes]
            )
            packed_attn_output = packed_attn_output_

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values
class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config=config
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttentionVL(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2VLMoTDecoderLayer(nn.Module):
    """Decoder layer with TokenAR (causal tokens) and BlockAR (latent-block) experts."""

    def __init__(
        self, 
        config, 
        layer_idx: Optional[int] = None, 
        attn_module: Optional[Qwen2VLAttention] = PackedAttentionMoTVL,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_token_ar = config.freeze_token_ar

        self.self_attn = attn_module(config, layer_idx)
        self.config=config
        self.mlp = Qwen2MLP(config)
        self.mlp_block_ar = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_block_ar = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_block_ar = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_token_ar_indexes: torch.LongTensor,
        packed_block_ar_indexes: torch.LongTensor,
        packed_all_timestep_embeds= None,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_token_ar_indexes] = self.input_layernorm(packed_sequence[packed_token_ar_indexes])
        packed_sequence_[packed_block_ar_indexes] = self.input_layernorm_block_ar(packed_sequence[packed_block_ar_indexes],t=packed_all_timestep_embeds[packed_block_ar_indexes])

        # Self Attention
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_token_ar_indexes=packed_token_ar_indexes,
            packed_block_ar_indexes=packed_block_ar_indexes,
        )
        if self.freeze_token_ar:
            packed_sequence_[packed_token_ar_indexes] = packed_sequence_[packed_token_ar_indexes].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_token_ar_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_token_ar_indexes])
        )
        if self.freeze_token_ar:
            packed_sequence_[packed_token_ar_indexes] = packed_sequence_[packed_token_ar_indexes].detach()
    
        packed_sequence_[packed_block_ar_indexes] = self.mlp_block_ar(
            self.post_attention_layernorm_block_ar(packed_sequence[packed_block_ar_indexes],t=packed_all_timestep_embeds[packed_block_ar_indexes])
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="token_ar",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        packed_all_timestep_embeds= None,
    ) -> BaseNavitOutputWithPast:
        residual = packed_query_sequence
        if mode == "token_ar":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "block_ar":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_block_ar(packed_query_sequence[packed_vae_token_indexes], t=packed_all_timestep_embeds[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "token_ar":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "block_ar":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_block_ar(packed_vae_query_sequence, t=packed_all_timestep_embeds[packed_vae_token_indexes]).to(torch.bfloat16)

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_block_ar(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values

Decoder_layer_dict_VL = {
    "Qwen2VLDecoderLayer": Qwen2VLDecoderLayer,  # packed decoder
    "Qwen2VLMoTDecoderLayer": partial(Qwen2VLMoTDecoderLayer, attn_module=PackedAttentionMoTVL)  # multimodal MoT
}

class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = 'Mo' in config.layer_module

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict_VL[config.layer_module]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.config=config

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_block_ar = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.architectures[0]=="Qwen3VLForConditionalGeneration":
            self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        else:
            self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.post_init()

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)
    
    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_token_ar_indexes: Optional[torch.LongTensor] = None,
        packed_block_ar_indexes: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        packed_all_timestep_embeds= None,
    ) -> torch.Tensor:
        if self.config.freeze_token_ar:
            packed_sequence[packed_token_ar_indexes] = packed_sequence[packed_token_ar_indexes].detach()
        # build multimodal RoPE
        packed_position_embeddings= self.rotary_emb(packed_sequence.unsqueeze(0), packed_position_ids)

        extra_inputs = {}
        if self.use_moe:
            assert packed_token_ar_indexes is not None
            if packed_block_ar_indexes is None:
                packed_block_ar_indexes = packed_token_ar_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_token_ar_indexes=packed_token_ar_indexes,
                packed_block_ar_indexes=packed_block_ar_indexes,
            )
        for layer_idx, decoder_layer in enumerate(self.layers):
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                packed_all_timestep_embeds= packed_all_timestep_embeds,
                **extra_inputs
            )
            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                packed_sequence = self._deepstack_process(
                    packed_sequence,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_token_ar_indexes] = self.norm(packed_sequence[packed_token_ar_indexes])
            if self.config.freeze_token_ar:
                packed_sequence_[packed_token_ar_indexes] = packed_sequence_[packed_token_ar_indexes].detach()
            packed_sequence_[packed_block_ar_indexes] = self.norm_block_ar(packed_sequence[packed_block_ar_indexes])
            return packed_sequence_
        else:
            return self.norm(packed_sequence)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="token_ar",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        packed_all_timestep_embeds= None,

    ) -> BaseNavitOutputWithPast:

        # multimodal RoPE
        cos, sin = self.rotary_emb(packed_query_sequence.unsqueeze(0), packed_query_position_ids) #torch.Size([1, seq_len, 1536]) torch.Size([3, 1, seq_len])
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'block_ar':
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_all_timestep_embeds= packed_all_timestep_embeds,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )
            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                packed_query_sequence = self._deepstack_process(
                    packed_query_sequence,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        if self.use_moe:
            if mode == "token_ar":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "block_ar":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_block_ar(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2VLModel(config)  # packed / MoT variant
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self):
        state = self.state_dict()
        for name, param in self.named_parameters():
            if "_block_ar" in name:
                original_name = name.replace("_block_ar", "")
                if original_name in state:
                    param.data.copy_(state[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_token_ar_indexes: Optional[torch.LongTensor] = None,
        packed_block_ar_indexes: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        packed_all_timestep_embeds: torch.Tensor = None,

    ) -> torch.Tensor:

        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_token_ar_indexes=packed_token_ar_indexes,
            packed_block_ar_indexes=packed_block_ar_indexes,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            packed_all_timestep_embeds=packed_all_timestep_embeds,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="token_ar",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        packed_all_timestep_embeds: torch.Tensor = None,
    ) -> BaseNavitOutputWithPast:

        outputs = self.model.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            packed_all_timestep_embeds=packed_all_timestep_embeds,
        )

        return outputs
