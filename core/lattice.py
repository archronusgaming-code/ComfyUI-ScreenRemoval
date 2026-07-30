"""
core/lattice.py

Estimate the screen lattice from paired FFT peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .pairing import PairingResult


@dataclass
class Lattice:
    origin: tuple[float, float]

    basis_u: np.ndarray
    basis_v: np.ndarray

    pitch_x: float
    pitch_y: float

    rotation: float

    confidence: float


def estimate_lattice(
    pairing: PairingResult,
    width: int,
    height: int,
) -> Lattice:
    """
    Estimate the fundamental lattice vectors.
    """

    cx = width / 2.0
    cy = height / 2.0

    theta = np.deg2rad(pairing.rotation)

    basis_u = np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    basis_v = np.array([
        -np.sin(theta),
        np.cos(theta),
    ])

    return Lattice(
        origin=(cx, cy),
        basis_u=basis_u,
        basis_v=basis_v,
        pitch_x=pairing.pitch_x,
        pitch_y=pairing.pitch_y,
        rotation=pairing.rotation,
        confidence=pairing.confidence,
    )