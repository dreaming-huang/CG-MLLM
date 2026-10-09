# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os


def get_parquet_data_paths(data_dir_list, num_sampled_data_paths):
    data_paths = []
    for data_dir, num_data_path in zip(data_dir_list, num_sampled_data_paths):
        data_paths_per_dir = sorted(
            os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(".parquet")
        )
        repeat = num_data_path // len(data_paths_per_dir)
        data_paths_per_dir = data_paths_per_dir * (repeat + 1)
        data_paths.extend(data_paths_per_dir[:num_data_path])
    return data_paths
