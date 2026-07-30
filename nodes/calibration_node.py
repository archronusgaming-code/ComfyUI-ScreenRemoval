from pathlib import Path

from ..core.fft import compute_fft
from ..core.profile import ScreenProfile
from ..core.image_utils import tensor_to_numpy, numpy_to_tensor


class ScreenCalibrationNode:

    CATEGORY = "Screen Removal"
    FUNCTION = "calibrate"

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "FLOAT",
        "FLOAT",
    )

    RETURN_NAMES = (
        "fft_preview",
        "profile_path",
        "pitch",
        "confidence",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def calibrate(self, image):

        # Convert Comfy tensor -> numpy
        img = tensor_to_numpy(image)

        # Compute FFT
        fft = compute_fft(img)

        # Create calibration profile
        profile = ScreenProfile(
            width=img.shape[1],
            height=img.shape[0],
        )

        profile_path = (
            Path(__file__).parent.parent
            / "profiles"
            / "default_profile.json"
        )

        profile.save(profile_path)

        # Convert preview back to Comfy IMAGE
        preview = numpy_to_tensor(fft.preview)

        return (
            preview,
            str(profile_path),
            profile.pitch_x,
            profile.confidence,
        )