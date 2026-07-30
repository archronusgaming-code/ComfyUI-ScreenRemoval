import cv2
import numpy as np


def draw_lattice(image, lattice, harmonics=10):

    out = image.copy()

    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    origin = np.array(lattice.origin)

    u = lattice.basis_u * lattice.pitch_x
    v = lattice.basis_v * lattice.pitch_y

    for i in range(-harmonics, harmonics + 1):
        for j in range(-harmonics, harmonics + 1):

            p = origin + i * u + j * v

            x = int(round(p[0]))
            y = int(round(p[1]))

            if (
                0 <= x < out.shape[1]
                and
                0 <= y < out.shape[0]
            ):
                cv2.circle(
                    out,
                    (x, y),
                    3,
                    (255, 255, 0),
                    -1,
                )

    return out