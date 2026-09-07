# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import numpy as np
import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    """Build a causal + parallel mask within one sample, while isolating noise tokens."""

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)
    document_id = torch.cat(
        [torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]
    ).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def patchify_qwenvl(image, patch_size, temporal_patch_size=2, merge_size=2):
    """Flatten a single-frame image into Qwen-VL style patches."""
    c, h, w = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Input size must be divisible by patch_size"

    patches = image.unsqueeze(0).numpy()
    if patches.shape[0] == 1:
        patches = np.tile(patches, (temporal_patch_size, 1, 1, 1))

    T, C, H, W = patches.shape
    grid_t = T // temporal_patch_size
    grid_h, grid_w = H // patch_size, W // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size * patch_size
    )
    return torch.from_numpy(flatten_patches)


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    return (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    return (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")
    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []
    for token in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|obj_start|>", "<|obj_end|>"):
        if token not in all_special_tokens:
            new_tokens.append(token)

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = dict(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        start_of_image=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        end_of_image=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        start_of_obj=tokenizer.convert_tokens_to_ids("<|obj_start|>"),
        end_of_obj=tokenizer.convert_tokens_to_ids("<|obj_end|>"),
    )
    return tokenizer, new_token_ids, num_new_tokens


def packed_to_mrope_position_ids(
    packed_position_ids,
    packed_vit_token_indexes,
    image_grid_thw,
    spatial_merge_size: int = 1,
    image_token_is_single_value: bool = True,
):
    """Convert packed 1D position ids to Qwen2-VL style (3, batch, seq_len) mRoPE ids."""
    packed_pos = packed_position_ids
    seq_len = packed_pos.shape[-1]
    vit_mask = torch.zeros(seq_len, dtype=torch.bool)
    vit_mask[packed_vit_token_indexes] = True
    vit_mask = [vit_mask]

    if packed_pos.ndim != 2:
        raise ValueError("packed_position_ids must be shape (batch, seq_len)")
    batch, seq_len = packed_pos.shape
    device = packed_pos.device
    dtype = packed_pos.dtype

    pos_3d = torch.zeros(3, batch, seq_len, dtype=dtype, device=device)
    mrope_deltas = torch.zeros(batch, 1, dtype=dtype, device=device)

    image_ptr = 0
    num_images_total = image_grid_thw.shape[0] if image_grid_thw is not None else 0

    for b in range(batch):
        one_pos = packed_pos[b]
        one_mask = vit_mask[b]
        i = 0
        max_pos_for_sample = -1
        while i < seq_len:
            if i == 0 or one_pos[i] == 0 or one_pos[i] < one_pos[i - 1]:
                offset = 0
            if one_mask[i].item():
                j = i + 1
                while j < seq_len and one_mask[j].item():
                    j += 1
                region_len = j - i

                if image_ptr >= num_images_total:
                    raise IndexError(f"image_grid_thw shortage: needed more images at sample {b} starting idx {i}")

                t = int(image_grid_thw[image_ptr, 0].item())
                h = int(image_grid_thw[image_ptr, 1].item()) // spatial_merge_size
                w = int(image_grid_thw[image_ptr, 2].item()) // spatial_merge_size
                image_ptr += 1

                num_patches = t * h * w
                if region_len != num_patches:
                    raise ValueError(
                        f"Mismatch in image patch count at batch {b} pos {i}:{j} - "
                        f"region_len={region_len}, grid patches={num_patches}"
                    )

                base = int(one_pos[i].item()) + offset
                t_idx = torch.arange(t, device=device).view(-1, 1, 1).expand(t, h, w).flatten()
                h_idx = torch.arange(h, device=device).view(1, -1, 1).expand(t, h, w).flatten()
                w_idx = torch.arange(w, device=device).view(1, 1, -1).expand(t, h, w).flatten()
                grid = torch.stack([t_idx, h_idx, w_idx], dim=0).to(dtype)
                pos_3d[:, b, i:j] = grid + base
                max_pos_for_sample = max(max_pos_for_sample, int((grid + base).max().item()))
                offset += (max(t, h, w) - 1)
                i = j
            else:
                j = i + 1
                while j < seq_len and not one_mask[j].item():
                    j += 1
                original_block = one_pos[i:j].to(dtype)
                pos_block = original_block + offset
                pos_3d[0, b, i:j] = pos_block
                pos_3d[1, b, i:j] = pos_block
                pos_3d[2, b, i:j] = pos_block
                max_pos_for_sample = max(max_pos_for_sample, int(pos_block.max().item()))
                i = j

        if max_pos_for_sample < 0:
            mrope_deltas[b, 0] = 0
        else:
            mrope_deltas[b, 0] = max_pos_for_sample + 1 - seq_len

    return pos_3d, mrope_deltas


def packed_to_mrope_position_ids_repeat(packed_position_ids):
    """Repeat 1D position ids three times when there is no image, matching the mRoPE layout."""
    if packed_position_ids.ndim != 2:
        raise ValueError("packed_position_ids must be shape (batch, seq_len)")

    batch, seq_len = packed_position_ids.shape
    device = packed_position_ids.device
    dtype = packed_position_ids.dtype
    pos_3d = packed_position_ids.unsqueeze(0).expand(3, batch, seq_len).contiguous()
    mrope_deltas = torch.zeros(batch, 1, dtype=dtype, device=device)
    return pos_3d, mrope_deltas
