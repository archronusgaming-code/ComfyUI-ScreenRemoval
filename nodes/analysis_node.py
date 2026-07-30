from ..core.image_utils import tensor_to_numpy, numpy_to_tensor

from ..core.fft import compute_fft
from ..core.peaks import detect_screen_peaks, draw_peaks
from ..core.pairing import pair_peaks, draw_pairs
from ..core.lattice import estimate_lattice
from ..core.notch import build_notch_mask

import cv2
import numpy as np


class ScreenAnalysisNode:

    CATEGORY = "Screen Removal"

    FUNCTION = "analyze"

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "STRING",
    )

    RETURN_NAMES = (
        "fft",
        "peaks",
        "pairs",
        "lattice",
        "mask",
        "stats",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def analyze(self, image):

        #
        # Convert to numpy
        #
        img = tensor_to_numpy(image)

        #
        # FFT
        #
        fft = compute_fft(img)

        #
        # Peak detection
        #
        detection = detect_screen_peaks(
            fft.preview[:, :, 0]
        )

        peak_image = draw_peaks(
            fft.preview[:, :, 0],
            detection,
        )

        #
        # Pair peaks
        #
        pairing = pair_peaks(
            detection,
            fft.preview.shape[1],
            fft.preview.shape[0],
        )

        pair_image = draw_pairs(
            fft.preview[:, :, 0],
            pairing,
        )

        #
        # Lattice
        #
        lattice = estimate_lattice(
            pairing,
            fft.preview.shape[1],
            fft.preview.shape[0],
        )

        lattice_image = pair_image.copy()

        #
        # Build notch mask
        #
        notch = build_notch_mask(
            lattice,
            fft.preview.shape[:2],
        )

        mask = (
            notch.mask * 255
        ).astype(np.uint8)

        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2RGB,
        )

        #
        # Statistics
        #
        stats = (
            f"Detected Peaks : {len(detection.peaks)}\n"
            f"Matched Pairs : {len(pairing.pairs)}\n"
            f"Pitch X : {pairing.pitch_x:.2f}\n"
            f"Pitch Y : {pairing.pitch_y:.2f}\n"
            f"Rotation : {pairing.rotation:.2f}\n"
            f"Confidence : {pairing.confidence:.2f}"
        )

        return (
            numpy_to_tensor(fft.preview),
            numpy_to_tensor(peak_image),
            numpy_to_tensor(pair_image),
            numpy_to_tensor(lattice_image),
            numpy_to_tensor(mask),
            stats,
        )