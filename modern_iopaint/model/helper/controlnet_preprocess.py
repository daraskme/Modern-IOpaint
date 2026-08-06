import cv2
import numpy as np
import torch
from PIL import Image


def make_canny_control_image(image: np.ndarray) -> Image.Image:
    """Build a three-channel Canny conditioning image with OpenCV only."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    canny_image = np.repeat(canny_image[:, :, None], 3, axis=2)
    return Image.fromarray(canny_image)


def make_inpaint_control_image(
    image: np.ndarray, mask: np.ndarray
) -> torch.Tensor:
    """Build the native Diffusers inpaint ControlNet conditioning tensor."""
    control_image = image.astype(np.float32) / 255.0
    control_image[mask[:, :, -1] > 128] = -1.0
    control_image = np.expand_dims(control_image, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(control_image)
