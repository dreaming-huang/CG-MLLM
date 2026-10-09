# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback

import numpy as np
import torch

from .distributed_iterable_dataset import DistributedIterableDataset
from .vlm_dataset import conversations_to_elements, read_jsonl_lines


def pc_norm(pc):
    """Center xyz and scale it into the unit sphere, as PointBERT expects; extra channels are kept."""
    xyz = pc[:, :3] - np.mean(pc[:, :3], axis=0)
    xyz = xyz / np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    return np.concatenate((xyz, pc[:, 3:]), axis=1)


class PointLLMJSONLIterableDataset(DistributedIterableDataset):
    """PointLLM jsonl: {"object_id": str, "conversations": [...]}, points at `<data_dir>/<object_id>_8192.npy`.

    latent_length is the number of PointBERT tokens per object (512 groups + cls).
    """

    def __init__(
        self, dataset_name, tokenizer, jsonl_path_list, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0, latent_length=513,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.latent_length = latent_length
        self.data_paths = read_jsonl_lines(
            jsonl_path_list, data_dir_list, num_used_data, shuffle_lines, shuffle_seed, self.rng
        )
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = self.data_status[worker_id] + 1 if self.data_status is not None else 0
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, point_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(data)
                    point_cloud = np.load(os.path.join(point_dir, f"{data_item['object_id']}_8192.npy"))
                except Exception:
                    traceback.print_exc()
                    continue

                point_tensor = torch.from_numpy(pc_norm(point_cloud).astype(np.float32))
                num_tokens = self.latent_length
                text_ids_list, sequence_plan = [], []
                for item in conversations_to_elements(data_item["conversations"], "<point>", "und_vae_obj", 1):
                    if item["type"] == "text":
                        text_ids = self.tokenizer.encode(item["text"])
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            sequence_plan.append({"type": "text", "enable_cfg": 0, "loss": item["has_loss"]})
                    else:
                        sequence_plan.append({"type": "und_vae_obj", "num_tokens": self.latent_length})

                if not any(item.get("loss", 0) for item in sequence_plan):
                    continue

                yield dict(
                    image_tensor_list=[],
                    surface_list=[point_tensor],
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
