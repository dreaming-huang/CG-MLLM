# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import random

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .data_utils import make_shape_vae_surface, pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths

Image.MAX_IMAGE_PIXELS = 20_000_000

VALID_COND_MODES = ("mixed", "text_only", "image_only")


def resolve_obj_cond(cond_mode, has_caption, text_only_prob, image_only_prob, text_and_image_prob):
    """Return (has_text, has_image), or None to skip the sample."""
    if cond_mode == "text_only":
        return (True, False) if has_caption else None
    if cond_mode == "image_only" or not has_caption:
        return False, True

    total = text_only_prob + image_only_prob + text_and_image_prob
    if total <= 0:
        return False, True
    p = random.random() * total
    if p < text_only_prob:
        return True, False
    if p < text_only_prob + image_only_prob:
        return False, True
    return True, True


class I2ObjIterableDataset(DistributedIterableDataset):
    """Parquet rows with `surface` (npz bytes holding `random_surface`), `image_list` and `captions`.

    cond_mode:
      - mixed: sample text-only / image-only / text+image with the *_prob fields
      - text_only: text-to-3D, samples without captions are skipped
      - image_only: image-to-3D
    """

    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        latent_length=512, pc_size=None,
        cond_mode="mixed", text_only_prob=0.5, image_only_prob=0.5, text_and_image_prob=0.0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        if cond_mode not in VALID_COND_MODES:
            raise ValueError(f"Unknown cond_mode={cond_mode!r}, expected one of {VALID_COND_MODES}")
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.latent_length = latent_length
        self.pc_size = pc_size or latent_length * 20
        self.cond_mode = cond_mode
        self.text_only_prob = float(text_only_prob)
        self.image_only_prob = float(image_only_prob)
        self.text_and_image_prob = float(text_and_image_prob)
        self.data_paths = get_parquet_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def load_surface(self, surface_bytes):
        random_surface = np.load(io.BytesIO(surface_bytes), allow_pickle=True)["random_surface"]
        ind = np.random.default_rng().choice(random_surface.shape[0], self.pc_size, replace=False)
        return make_shape_vae_surface(random_surface[ind])

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride
        merge_size = self.transform.merge_size

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )
        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                with open(parquet_file_path, "rb") as f:
                    fr = pq.ParquetFile(f)
                    for row_group_id in range(row_group_start_id, fr.num_row_groups):
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            try:
                                surface = self.load_surface(row["surface"])
                            except Exception as e:
                                print(f"Error: {e} in rg#{row_group_id}, {parquet_file_path}")
                                continue

                            caption_list = row.get("captions", [])
                            if isinstance(caption_list, str):
                                try:
                                    caption_list = json.loads(caption_list)
                                except json.JSONDecodeError:
                                    caption_list = []
                            if caption_list is None or isinstance(caption_list, str):
                                caption_list = []
                            caps_token = [self.tokenizer.encode(v) for v in caption_list]

                            cond = resolve_obj_cond(
                                self.cond_mode, len(caps_token) > 0,
                                self.text_only_prob, self.image_only_prob, self.text_and_image_prob,
                            )
                            if cond is None:
                                continue
                            has_text, has_image = cond

                            num_tokens = self.latent_length
                            sequence_plan, text_ids_list, image_tensor_list = [], [], []
                            if has_text:
                                text_ids = random.choice(caps_token)
                                text_ids_list.append(text_ids)
                                num_tokens += len(text_ids)
                                sequence_plan.append({"type": "text", "enable_cfg": 1, "loss": 0})
                            if has_image:
                                try:
                                    image_bytes = random.choice(row["image_list"])
                                    image = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
                                except Exception as e:
                                    print(f"Error: {e} in rg#{row_group_id}, {parquet_file_path}")
                                    continue
                                image_tensor = self.transform(image)
                                height, width = image_tensor.shape[1:]
                                image_tensor_list.append(image_tensor)
                                num_tokens += width * height // (transform_stride ** 2 * merge_size ** 2)
                                sequence_plan.append({"type": "vit_image", "enable_cfg": 1, "loss": 0})
                            sequence_plan.append({"type": "vae_obj", "loss": 1, "num_tokens": self.latent_length})

                            yield dict(
                                image_tensor_list=image_tensor_list,
                                surface_list=[surface],
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                },
                            )
                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
