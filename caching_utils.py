from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import os
import shutil
import numba as nb
from loguru import logger
from pathlib import Path
from scored_storage import ScoredStorage
import math
import torch


nf4 = np.asarray(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)



@nb.jit
def make_feat_data(sae_indices_img, sae_weights_img, width, step, batch_size, img_seq_len, k, use_img):
    total_activations = sae_indices_img.size
    nums = np.empty(total_activations, dtype=np.uint32)
    indices = np.empty((total_activations, 3), dtype=np.uint32)
    activations = np.empty(total_activations, dtype=np.float32)
    index = 0
    for i in range(batch_size):
        if use_img:
            batch_features = sae_indices_img[i].ravel()
            batch_weights = sae_weights_img[i].ravel()
            unique_features = np.unique(batch_features)

            for feature_num in unique_features:
                # Find max activation for this feature
                mask = batch_features == feature_num
                max_idx = np.argmax(batch_weights[mask])

                # Map 1D index back to 2D coordinates
                flat_idx = np.flatnonzero(mask)[max_idx]
                # x, a = np.unravel_index(flat_idx, (img_seq_len, k))
                x, a = flat_idx // k, flat_idx % k
                h, w = x // width, x % width

                nums[index] = feature_num
                indices[index] = (step * batch_size + i, h, w)
                activations[index] = batch_weights[flat_idx]
                index += 1
    return nums[:index], indices[:index], activations[:index]


def save_itda_outputs(
        itda_weights_img: torch.Tensor,
        itda_indices_img: torch.Tensor,
        prompts: list[str], 
        images: np.ndarray,
        step: int,
        feature_acts: ScoredStorage,
        save_image_activations: bool,
        images_dir: Path,
        image_activations_dir: Path,
    ):
        image_max = 6.0
        images_dir.mkdir(parents=True, exist_ok=True)
        image_activations_dir.mkdir(parents=True, exist_ok=True)


        batch_size = itda_weights_img.shape[0]
        assert itda_weights_img.shape[0] == itda_indices_img.shape[0]
        assert batch_size == len(prompts) == images.shape[0]
        img_seq_len = itda_weights_img.shape[1]
        k = itda_weights_img.shape[2]
        assert k == itda_weights_img.shape[2] == itda_indices_img.shape[2]
        use_img = img_seq_len > 0
        if use_img:
            assert img_seq_len == itda_indices_img.shape[1] == images.shape[-1] * images.shape[-2] // 4
            images_to_save = (images / image_max).clip(-1, 1)
            images_to_save = np.abs(images_to_save[..., None] - nf4).argmin(-1).astype(np.uint8)
            images_to_save = (
                (images_to_save[..., ::2] & 0x0F)
                | ((images_to_save[..., 1::2] << 4) & 0xF0))
            for i, img in enumerate(images_to_save):
                np.savez(
                    images_dir / f"{step * batch_size + i}.npz",
                    img
                )
        width = images.shape[-1] // 2
        nums, indices, activations = make_feat_data(itda_indices_img, itda_weights_img, width, step, batch_size, img_seq_len, k, use_img)
        feature_acts.insert_many(nums, indices, activations)

        rows, _scores, mask = feature_acts.all_rows()
        used_rows = rows[mask].astype(np.uint64)
        unique_idces = np.unique(used_rows[:, 0])
        extant_images = set(unique_idces.tolist())

        if save_image_activations:
            image_activations_dir.mkdir(parents=True, exist_ok=True)
            for image in image_activations_dir.glob("*.npz"):
                identifier = int(image.stem)
                if identifier not in extant_images:
                    image.unlink()
            for i in range(batch_size):
                identifier = step * batch_size + i
                if identifier not in extant_images:
                    continue
                np.savez(
                    image_activations_dir / f"{identifier}.npz",
                    itda_indices_img[i],
                    itda_weights_img[i],
                )