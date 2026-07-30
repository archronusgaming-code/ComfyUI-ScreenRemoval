"""
core/fft.py

FFT processing for periodic screen detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FFTData:
    """
    Container for FFT processing results.
    """

    gray: np.ndarray
    window: np.ndarray
    windowed: np.ndarray

    spectrum: np.ndarray

    magnitude: np.ndarray

    preview: np.ndarray


def apply_hann_window(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a 2D Hann window to reduce edge artifacts.
    """

    h, w = gray.shape

    window = np.outer(
        np.hanning(h),
        np.hanning(w),
    ).astype(np.float32)

    windowed = gray * window

    return window, windowed


def compute_fft(image: np.ndarray) -> FFTData:
    """
    Compute centered FFT and preview image.
    """

    #
    # Convert to grayscale
    #

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32)

    else:

        gray = image.astype(np.float32)

    #
    # Hann window
    #

    window, windowed = apply_hann_window(gray)

    #
    # FFT
    #

    spectrum = np.fft.fft2(windowed)

    spectrum = np.fft.fftshift(spectrum)

    #
    # Magnitude
    #

    magnitude = np.abs(spectrum)

    #
    # Log magnitude
    #

    magnitude = np.log1p(magnitude)

    #
    # Remove bright DC spike for display only
    #

    cy = magnitude.shape[0] // 2
    cx = magnitude.shape[1] // 2

    preview = magnitude.copy()

    preview[
        cy - 2 : cy + 3,
        cx - 2 : cx + 3,
    ] = np.median(preview)

    #
    # Normalize preview
    #

    preview = cv2.normalize(
        preview,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    preview = preview.astype(np.uint8)

    preview = cv2.cvtColor(
        preview,
        cv2.COLOR_GRAY2RGB,
    )

    return FFTData(
        gray=gray,
        window=window,
        windowed=windowed,
        spectrum=spectrum,
        magnitude=magnitude,
        preview=preview,
    )


def inverse_fft(
    spectrum: np.ndarray,
) -> np.ndarray:
    """
    Convert a centered FFT spectrum back into
    a spatial image.
    """

    shifted = np.fft.ifftshift(spectrum)

    image = np.fft.ifft2(shifted)

    image = np.real(image)

    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return image.astype(np.uint8)