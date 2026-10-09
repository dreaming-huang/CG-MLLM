import io
import math
import os
import random
import tarfile

import numpy as np
from PIL import Image

from .data_utils import make_shape_vae_surface, pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 20_000_000

POINTS_PER_GROUP = 1024


class HY3DFullIterableDataset(DistributedIterableDataset):
    """HY3D-Bench image-to-3D. Each data_dir (e.g. HY3D-Bench/full/train) contains:
        images/chunk_xxxx/<uid>_vc_render.npz       rendered views (png bytes)
        sample_points/chunk_xxxx/*_surface_*.tar    `<chunk>_<uid>.random_surface.npy` point groups
    num_used_data is the number of chunks used per data_dir.
    """

    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        latent_length=4096, pc_size=None,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.latent_length = latent_length
        self.pc_size = pc_size or latent_length * 20
        self._chunk_surface_tars = {}
        self.data_paths = []
        for data_root, num_chunks in zip(data_dir_list, num_used_data):
            self.data_paths.extend(self._collect_samples(data_root, num_chunks))
        self.set_epoch()

    def _collect_samples(self, data_root, num_chunks):
        image_root = os.path.join(data_root, "images")
        points_root = os.path.join(data_root, "sample_points")
        image_chunks = {n for n in os.listdir(image_root) if n.startswith("chunk_")}
        points_chunks = {n for n in os.listdir(points_root) if n.startswith("chunk_")}
        chunks = sorted(image_chunks & points_chunks)
        if len(chunks) == 0:
            raise ValueError(f"No valid HY3D chunks found under {data_root}")
        chunks = (chunks * (num_chunks // len(chunks) + 1))[:num_chunks]

        samples = []
        for chunk_name in chunks:
            points_chunk_dir = os.path.join(points_root, chunk_name)
            surface_tars = sorted(
                os.path.join(points_chunk_dir, n)
                for n in os.listdir(points_chunk_dir)
                if n.endswith(".tar") and "_surface_" in n and "_sharpedge_surface_" not in n
            )
            if len(surface_tars) == 0:
                continue
            self._chunk_surface_tars[points_chunk_dir] = surface_tars
            for uid, image_path in self._build_uid_image_map(os.path.join(image_root, chunk_name)).items():
                samples.append((points_chunk_dir, uid, f"{chunk_name}_{uid}", image_path))

        if len(samples) == 0:
            raise ValueError(f"No valid HY3D samples found under {data_root}")
        return samples

    @staticmethod
    def _build_uid_image_map(images_chunk_dir):
        """Prefer `<uid>_vc_render.npz` over `<uid>_vc_complete_render.npz`."""
        uid_image_paths, uid_priority = {}, {}
        for name in os.listdir(images_chunk_dir):
            if name.endswith("_vc_render.npz"):
                uid, priority = name[: -len("_vc_render.npz")], 2
            elif name.endswith("_vc_complete_render.npz"):
                uid, priority = name[: -len("_vc_complete_render.npz")], 1
            else:
                continue
            if priority > uid_priority.get(uid, -1):
                uid_priority[uid] = priority
                uid_image_paths[uid] = os.path.join(images_chunk_dir, name)
        return uid_image_paths

    def _load_image_tensor(self, image_npz_path):
        image_npz = np.load(image_npz_path, allow_pickle=True)
        image_keys = [
            k for k in image_npz.files
            if k.endswith(".png") and "_camnorm" not in k and "_basecolor" not in k
        ]
        if len(image_keys) == 0:
            return None
        image = pil_img2rgb(Image.open(io.BytesIO(image_npz[random.choice(image_keys)].tobytes())))
        return self.transform(image)

    def _load_surface(self, points_chunk_dir, key):
        tar_paths = self._chunk_surface_tars[points_chunk_dir]
        rng = np.random.default_rng()
        num_groups = max(1, math.ceil(self.pc_size / POINTS_PER_GROUP))
        group_ids = rng.choice(len(tar_paths), size=num_groups, replace=len(tar_paths) < num_groups)
        member_name = f"{key}.random_surface.npy"

        group_points = []
        for group_id in group_ids:
            with tarfile.open(tar_paths[int(group_id)], "r") as tf:
                try:
                    f = tf.extractfile(tf.getmember(member_name))
                except KeyError:
                    continue
                if f is not None:
                    group_points.append(np.load(io.BytesIO(f.read())).astype(np.float32, copy=False))
        if len(group_points) == 0:
            raise ValueError(f"No points found for {member_name}")

        points = np.concatenate(group_points, axis=0)
        ids = rng.choice(points.shape[0], size=self.pc_size, replace=points.shape[0] < self.pc_size)
        return make_shape_vae_surface(points[ids])

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        sample_start_id = self.data_status[worker_id][0] + 1 if self.data_status is not None else 0
        transform_stride = self.transform.stride
        merge_size = self.transform.merge_size
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at sample#{sample_start_id}"
        )

        while True:
            worker_data_paths = data_paths_per_worker[sample_start_id:]
            for sample_idx, (points_chunk_dir, uid, key, image_npz_path) in enumerate(
                worker_data_paths, start=sample_start_id
            ):
                try:
                    image_tensor = self._load_image_tensor(image_npz_path)
                    if image_tensor is None:
                        continue
                    surface = self._load_surface(points_chunk_dir, key)
                except Exception as e:
                    print(f"HY3D load error: {e} in {key}")
                    continue

                height, width = image_tensor.shape[1:]
                image_length = width * height // (transform_stride ** 2 * merge_size ** 2)
                yield dict(
                    image_tensor_list=[image_tensor],
                    surface_list=[surface],
                    text_ids_list=[],
                    num_tokens=image_length + self.latent_length,
                    sequence_plan=[
                        {"type": "vit_image", "enable_cfg": 1, "loss": 0},
                        {"type": "vae_obj", "loss": 1, "num_tokens": self.latent_length},
                    ],
                    data_indexes={
                        "data_indexes": [sample_idx, 0, 0],
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                )

            sample_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
