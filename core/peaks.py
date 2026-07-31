from dataclasses import dataclass

import cv2
import numpy as np


# ------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------

@dataclass
class PeakDetection:
    pitch_x: float
    pitch_y: float
    rotation: float
    confidence: float
    peaks: list[tuple[int, int]]

# ------------------------------------------------------------------------
# Peak Detection
# ------------------------------------------------------------------------

def detect_screen_peaks(fft_image):
    """
    Detect bright FFT candidate peaks.

    This stage intentionally returns many candidates.
    Symmetry filtering is performed later by pairing.py.
    """

    img = fft_image.astype(np.uint8).copy()

    h, w = img.shape
    cx = w // 2
    cy = h // 2

    # ------------------------------------------------------------
    # Remove DC spike
    # ------------------------------------------------------------

    cv2.circle(img, (cx, cy), 18, 0, -1)

    # ------------------------------------------------------------
    # Blur slightly
    # ------------------------------------------------------------

    img = cv2.GaussianBlur(img, (5, 5), 0)

    # ------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------

    median = np.median(img)
    mad = np.median(np.abs(img - median))

    threshold = max(
        np.percentile(img, 99.4),
        median + (5.0 * mad),
    )

    binary = np.zeros_like(img)
    binary[img >= threshold] = 255

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.dilate(
        binary,
        kernel,
        iterations=1,
    )

    CENTER_RADIUS = 28
    AXIS_WIDTH = 8

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    peaks = []

    for i in range(1, nlabels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < 2:
            continue

        if area > 300:
            continue

        x = int(round(centroids[i][0]))
        y = int(round(centroids[i][1]))

        dx = x - cx
        dy = y - cy

        if np.hypot(dx, dy) < CENTER_RADIUS:
            continue

        if abs(dx) < AXIS_WIDTH:
            continue

        if abs(dy) < AXIS_WIDTH:
            continue

        peaks.append((x, y))

    peaks.sort(
        key=lambda p: np.hypot(
            p[0] - cx,
            p[1] - cy,
        )
    )

    confidence = min(
        len(peaks) / 40.0,
        1.0,
    )

    return PeakDetection(
        pitch_x=0.0,
        pitch_y=0.0,
        rotation=0.0,
        confidence=confidence,
        peaks=peaks,
    )
# ------------------------------------------------------------------------
# Debug Drawing
# ------------------------------------------------------------------------

def draw_peaks(fft_image, detection):

    """
    Draw detected FFT peaks.
    """

    if fft_image.ndim == 2:
        out = cv2.cvtColor(
            fft_image,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        out = fft_image.copy()

    h, w = out.shape[:2]

    cx = w // 2
    cy = h // 2

    #
    # Center
    #
    cv2.drawMarker(
        out,
        (cx, cy),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        18,
        2,
    )

    #
    # Peaks
    #
    for x, y in detection.peaks:

        cv2.circle(
            out,
            (x, y),
            6,
            (0, 0, 255),
            2,
        )

    #
    # Statistics
    #
    cv2.putText(
        out,
        f"Peaks: {len(detection.peaks)}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        out,
        f"Confidence: {detection.confidence:.2f}",
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    return out