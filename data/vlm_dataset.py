# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback

from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 2 ** 20


def read_jsonl_lines(jsonl_path_list, data_dir_list, num_used_data, shuffle_lines, shuffle_seed, rng):
    data_paths = []
    for jsonl_path, data_dir, num_data_point in zip(jsonl_path_list, data_dir_list, num_used_data):
        with open(jsonl_path, "r") as f:
            raw_data = f.readlines()
        if shuffle_lines:
            rng.seed(shuffle_seed)
            rng.shuffle(raw_data)
        data_paths.extend([(line, data_dir) for line in raw_data[:num_data_point]])
    return data_paths


def conversations_to_elements(conversations, placeholder, element_type, num_items):
    """Split LLaVA-style conversations into text / media elements; only `gpt` turns carry loss."""
    elements = []
    num_placeholders = sum(c["value"].count(placeholder) for c in conversations if c["from"] == "human")
    for _ in range(num_items - num_placeholders):
        elements.append({"type": element_type})

    for conversation in conversations:
        if conversation["from"] == "human" and placeholder not in conversation["value"]:
            elements.append({"type": "text", "has_loss": 0, "text": conversation["value"]})
        elif conversation["from"] == "human":
            text_list = conversation["value"].split(placeholder)
            for idx, text in enumerate(text_list):
                if text.strip() != "":
                    elements.append({"type": "text", "has_loss": 0, "text": text.strip()})
                if idx != len(text_list) - 1 and idx < num_items:
                    elements.append({"type": element_type})
        elif conversation["from"] == "gpt":
            elements.append({"type": "text", "has_loss": 1, "text": conversation["value"]})
    return elements


class SftJSONLIterableDataset(DistributedIterableDataset):
    """LLaVA-style jsonl: {"image": str | [str], "conversations": [{"from": "human"|"gpt", "value": ...}]}."""

    def __init__(
        self, dataset_name, transform, tokenizer, jsonl_path_list, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = read_jsonl_lines(
            jsonl_path_list, data_dir_list, num_used_data, shuffle_lines, shuffle_seed, self.rng
        )
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = self.data_status[worker_id] + 1 if self.data_status is not None else 0
        transform_stride = self.transform.stride
        merge_size = self.transform.merge_size
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(data)
                    image_names = data_item.get("image", [])
                    if isinstance(image_names, str):
                        image_names = [image_names]
                    raw_images = [pil_img2rgb(Image.open(os.path.join(image_dir, n))) for n in image_names]
                except Exception:
                    traceback.print_exc()
                    continue

                num_tokens = 0
                image_tensor_list = []
                for raw_image in raw_images:
                    image_tensor = self.transform(raw_image, img_num=len(raw_images))
                    image_tensor_list.append(image_tensor)
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // (transform_stride ** 2 * merge_size ** 2)

                text_ids_list, sequence_plan = [], []
                for item in conversations_to_elements(
                    data_item["conversations"], "<image>", "vit_image", len(image_tensor_list)
                ):
                    if item["type"] == "text":
                        text_ids = self.tokenizer.encode(item["text"])
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            sequence_plan.append({"type": "text", "enable_cfg": 0, "loss": item["has_loss"]})
                    else:
                        sequence_plan.append({"type": "vit_image", "enable_cfg": 0, "loss": 0})

                if not any(item["loss"] for item in sequence_plan):
                    continue

                yield dict(
                    image_tensor_list=image_tensor_list,
                    surface_list=[],
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
