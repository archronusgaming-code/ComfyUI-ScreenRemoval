"""
core/remove.py
"""

from __future__ import annotations

import cv2
import numpy as np


def inverse_fft(spectrum):

    ishift = np.fft.ifftshift(spectrum)

    image = np.fft.ifft2(ishift)

    image = np.abs(image)

    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return image.astype(np.uint8)


def apply_notch_filter(
    fft_data,
    notch,
):

    filtered = fft_data.spectrum * notch.mask

    image = inverse_fft(filtered)

    return image