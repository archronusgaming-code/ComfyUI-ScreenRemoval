"""
core/pairing.py

Symmetric FFT peak pairing and lattice estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .peaks import PeakDetection


@dataclass
class PeakPair:
    p1: tuple[int, int]
    p2: tuple[int, int]

    distance: float
    angle: float

    error: float


@dataclass
class PairingResult:
    pairs: list[PeakPair]

    pitch_x: float
    pitch_y: float

    rotation: float

    confidence: float


def pair_peaks(
    detection: PeakDetection,
    width: int,
    height: int,
    tolerance: float = 6.0,
) -> PairingResult:
    """
    Pair FFT peaks that are symmetric about the center.

    Returns:
        PairingResult
    """

    cx = width / 2.0
    cy = height / 2.0

    used = set()

    pairs = []

    for i, (x1, y1) in enumerate(detection.peaks):

        if i in used:
            continue

        target_x = 2.0 * cx - x1
        target_y = 2.0 * cy - y1

        best_index = None
        best_error = float("inf")

        for j, (x2, y2) in enumerate(detection.peaks):

            if j == i or j in used:
                continue

            error = math.hypot(
                target_x - x2,
                target_y - y2,
            )

            if error < best_error:
                best_error = error
                best_index = j

        if best_index is None:
            continue

        if best_error > tolerance:
            continue

        x2, y2 = detection.peaks[best_index]

        dx = x1 - cx
        dy = y1 - cy

        distance = math.hypot(dx, dy)

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        pairs.append(
            PeakPair(
                p1=(x1, y1),
                p2=(x2, y2),
                distance=distance,
                angle=angle,
                error=best_error,
            )
        )

        used.add(i)
        used.add(best_index)

    #
    # Estimate horizontal / vertical pitch
    #

    horizontal = []
    vertical = []
    rotations = []

    for pair in pairs:

        rotations.append(pair.angle)

        if abs(math.cos(math.radians(pair.angle))) > abs(
            math.sin(math.radians(pair.angle))
        ):
            horizontal.append(pair.distance)
        else:
            vertical.append(pair.distance)

    pitch_x = 0.0
    pitch_y = 0.0

    if horizontal:
        pitch_x = width / (2.0 * np.mean(horizontal))

    if vertical:
        pitch_y = height / (2.0 * np.mean(vertical))

    #
    # Circular mean for rotation
    #

    if rotations:

        angles = np.radians(rotations)

        rotation = np.degrees(
            np.arctan2(
                np.mean(np.sin(angles)),
                np.mean(np.cos(angles)),
            )
        )

    else:

        rotation = 0.0

    #
    # Confidence
    #

    if len(detection.peaks) == 0:

        confidence = 0.0

    else:

        confidence = len(pairs) / len(detection.peaks)

    confidence = float(
        np.clip(confidence, 0.0, 1.0)
    )

    return PairingResult(
        pairs=pairs,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        rotation=rotation,
        confidence=confidence,
    )


def draw_pairs(
    fft_image: np.ndarray,
    result: PairingResult,
) -> np.ndarray:
    """
    Draw paired FFT peaks for debugging.
    """

    out = cv2.cvtColor(
        fft_image,
        cv2.COLOR_GRAY2BGR,
    )

    for pair in result.pairs:

        cv2.line(
            out,
            pair.p1,
            pair.p2,
            (0, 255, 0),
            1,
        )

        cv2.circle(
            out,
            pair.p1,
            5,
            (0, 0, 255),
            2,
        )

        cv2.circle(
            out,
            pair.p2,
            5,
            (255, 0, 0),
            2,
        )

    return out