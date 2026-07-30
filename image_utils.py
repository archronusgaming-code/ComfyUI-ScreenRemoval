import numpy as np
import torch


def tensor_to_numpy(image):
    """ComfyUI IMAGE -> RGB uint8 numpy"""

    if image.ndim == 4:
        image = image[0]

    image = image.cpu().numpy()

    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    return image


def numpy_to_tensor(image):
    """RGB uint8 numpy -> ComfyUI IMAGE"""

    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    image = np.ascontiguousarray(image)

    return torch.from_numpy(image).unsqueeze(0).contiguous()