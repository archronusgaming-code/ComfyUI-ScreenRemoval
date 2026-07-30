"""
core/notch.py

Gaussian notch filter generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import Lattice


@dataclass
class NotchMask:
    mask: np.ndarray
    peaks: list[tuple[int, int]]


def gaussian_notch(shape, center, sigma=4.0, depth=0.95):

    h, w = shape

    y, x = np.ogrid[:h, :w]

    dist2 = (
        (x - center[0]) ** 2 +
        (y - center[1]) ** 2
    )

    return 1.0 - depth * np.exp(
        -dist2 / (2 * sigma * sigma)
    )


def build_notch_mask(
    lattice: Lattice,
    shape,
    harmonics=12,
    sigma=4.0,
    depth=0.95,
):

    h, w = shape

    mask = np.ones((h, w), dtype=np.float32)

    peaks = []

    origin = np.array(lattice.origin)

    u = lattice.basis_u * lattice.pitch_x
    v = lattice.basis_v * lattice.pitch_y

    for i in range(-harmonics, harmonics + 1):
        for j in range(-harmonics, harmonics + 1):

            if i == 0 and j == 0:
                continue

            p = origin + i * u + j * v

            x = int(round(p[0]))
            y = int(round(p[1]))

            if x < 0 or x >= w:
                continue

            if y < 0 or y >= h:
                continue

            peaks.append((x, y))

            mask *= gaussian_notch(
                (h, w),
                (x, y),
                sigma=sigma,
                depth=depth,
            )

    return NotchMask(
        mask=mask,
        peaks=peaks,
    )