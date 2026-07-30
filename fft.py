"""
core/fft.py

FFT processing utilities for ComfyUI-ScreenRemoval
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FFTData:
    """Container for FFT results."""

    gray: np.ndarray
    windowed: np.ndarray
    spectrum: np.ndarray
    magnitude: np.ndarray          # Original magnitude (keep untouched)
    preview: np.ndarray            # RGB preview for ComfyUI


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale float32."""

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return gray.astype(np.float32)


def apply_hann_window(gray: np.ndarray) -> np.ndarray:
    """Apply a 2D Hann window before the FFT."""

    h, w = gray.shape

    window = np.outer(
        np.hanning(h),
        np.hanning(w),
    )

    return gray * window


def remove_dc_peak(
    magnitude: np.ndarray,
    radius: int = 12,
) -> np.ndarray:
    """
    Zero the bright DC component for display/peak detection.
    """

    result = magnitude.copy()

    h, w = result.shape

    cy = h // 2
    cx = w // 2

    y, x = np.ogrid[:h, :w]

    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    result[mask] = 0

    return result


def normalize_preview(
    magnitude: np.ndarray,
) -> np.ndarray:
    """Convert magnitude image to an RGB preview."""

    preview = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    preview = preview.astype(np.uint8)

    return cv2.cvtColor(
        preview,
        cv2.COLOR_GRAY2RGB,
    )


def compute_fft(image: np.ndarray) -> FFTData:
    """
    Compute centered FFT and generate a display preview.
    """

    gray = rgb_to_gray(image)

    windowed = apply_hann_window(gray)

    spectrum = np.fft.fftshift(
        np.fft.fft2(windowed)
    )

    magnitude = np.log1p(
        np.abs(spectrum)
    )

    # Only the preview gets the center removed.
    preview_mag = remove_dc_peak(
        magnitude,
        radius=12,
    )

    preview = normalize_preview(
        preview_mag
    )

    return FFTData(
        gray=gray,
        windowed=windowed,
        spectrum=spectrum,
        magnitude=magnitude,
        preview=preview,
    )