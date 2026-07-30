"""
core/image_utils.py

Conversion helpers between ComfyUI tensors and OpenCV/NumPy images.
"""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    Convert a ComfyUI IMAGE tensor into a uint8 RGB NumPy array.

    Input:
        (1, H, W, 3) float32 in range [0, 1]

    Output:
        (H, W, 3) uint8 in range [0, 255]
    """

    if image.ndim != 4:
        raise ValueError(
            f"Expected IMAGE tensor with 4 dimensions, got {image.shape}"
        )

    image = image[0]

    image = image.detach().cpu().numpy()

    image = np.clip(image, 0.0, 1.0)

    image = (image * 255.0).round().astype(np.uint8)

    return np.ascontiguousarray(image)


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image into a ComfyUI IMAGE tensor.

    Accepts:
        uint8
        float32

    Returns:
        (1, H, W, 3) float32
    """

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    if image.max() > 1.0:
        image /= 255.0

    image = np.clip(image, 0.0, 1.0)

    tensor = torch.from_numpy(
        np.ascontiguousarray(image)
    )

    return tensor.unsqueeze(0)


def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8 image to float32 grayscale.
    """

    if image.ndim == 2:
        gray = image

    else:
        import cv2

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    return gray.astype(np.float32)