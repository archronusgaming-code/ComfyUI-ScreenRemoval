"""
image_utils.py

Shared image conversion utilities for ComfyUI-ScreenRemoval.

Author: ComfyUI-ScreenRemoval
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Torch <-> NumPy
# -----------------------------------------------------------------------------

def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    Convert a ComfyUI IMAGE tensor to uint8 RGB.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected torch.Tensor")

    image = image.detach().cpu()

    if image.ndim == 4:
        image = image[0]

    img = image.numpy()

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    return img


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert an OpenCV image into a ComfyUI IMAGE tensor.

    Supports:
        uint8
        float32
        grayscale
        RGB
    """

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    image = np.clip(image, 0.0, 1.0)

    return torch.from_numpy(image).unsqueeze(0)


# -----------------------------------------------------------------------------
# Color
# -----------------------------------------------------------------------------

def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB/BGR image to grayscale.
    """

    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is RGB.
    """

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


# -----------------------------------------------------------------------------
# FFT Helpers
# -----------------------------------------------------------------------------

def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for preview.
    """

    image = image.astype(np.float32)

    mn = image.min()
    mx = image.max()

    if mx <= mn:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - mn) / (mx - mn)

    return (image * 255).astype(np.uint8)


def log_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """
    Log scale FFT magnitude.
    """

    return np.log1p(magnitude)


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def draw_cross(
    image: np.ndarray,
    center,
    color=(0, 255, 255),
    size=12,
    thickness=2,
):
    """
    Draw crosshair.
    """

    cv2.drawMarker(
        image,
        center,
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=size,
        thickness=thickness,
    )

    return image


def draw_text(
    image: np.ndarray,
    text: str,
    position,
    color=(0, 255, 0),
):
    """
    Draw debug text.
    """

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert any image to uint8.
    """

    if image.dtype == np.uint8:
        return image

    image = normalize_for_display(image)

    return image


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def is_grayscale(image: np.ndarray) -> bool:
    return image.ndim == 2


def image_center(image: np.ndarray):
    """
    Return image center.
    """

    h, w = image.shape[:2]

    return (w // 2, h // 2)