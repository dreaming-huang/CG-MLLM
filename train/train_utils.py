# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


def create_logger(logging_dir, rank, filename="log"):
    """Log to stdout and `<logging_dir>/<filename>.txt` on rank 0; a no-op logger elsewhere."""
    logger = logging.getLogger(__name__)
    if rank == 0 and logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/{filename}.txt"),
            ],
        )
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    step_dirs = [d for d in os.listdir(checkpoint_dir) if d.isdigit() and os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    return os.path.join(checkpoint_dir, max(step_dirs, key=int))
