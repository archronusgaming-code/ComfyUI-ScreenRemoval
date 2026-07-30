from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PeakDetection:
    pitch_x: float
    pitch_y: float
    rotation: float
    confidence: float
    peaks: list[tuple[int, int]]


def detect_screen_peaks(fft_image):

    img = fft_image.copy().astype(np.uint8)

    h, w = img.shape

    cx = w // 2
    cy = h // 2

    #
    # remove DC area
    #
    cv2.circle(img, (cx, cy), 20, 0, -1)

    #
    # only brightest frequencies
    #
    thresh = np.percentile(img, 99.8)

    binary = np.zeros_like(img)

    binary[img >= thresh] = 255

    #
    # clean tiny noise
    #
    kernel = np.ones((3,3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    #
    # connected components
    #
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    peaks = []

    for i in range(1, nlabels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < 2:
            continue

        x = int(centroids[i][0])
        y = int(centroids[i][1])

        peaks.append((x, y))

        return PeakDetection(
        pitch_x=0,
        pitch_y=0,
        rotation=0,
        confidence=len(peaks),
        peaks=peaks
    )


def draw_peaks(fft_image, detection):

    out = cv2.cvtColor(fft_image, cv2.COLOR_GRAY2BGR)

    for x, y in detection.peaks:
        cv2.circle(out, (x, y), 6, (0, 0, 255), 2)

    return out