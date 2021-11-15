"""Corrupt images for robustness testing."""
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomRotation


def jpeg_compression(image: Image) -> Image:
    """Compute a compressed version of the input image.

    Args:
        image (Image): The input image.
        jpeg_compression (int): Compression factor
          on a scale from 0 (worst) to 95 (best).

    Returns:
        Image: The compressed image.
    """
    out = BytesIO()
    factor = np.random.randint(low=10, high=95)
    image.save(out, format="JPEG", quality=factor)
    return Image.open(out)


def random_rotation(image: Image, angle=15) -> Image:
    """Randomly rotates an Image.

    Args:
        image (Image): The input image.
        angle (int, optional): The max rotation angle. Defaults to 15.

    Returns:
        Image: The rotated image.
    """
    return RandomRotation(angle)(image)


def random_resized_crop(image: Image) -> Image:
    """Randomly resize and crop the input Image.

    Args:
        image (Image): The input image.

    Returns:
        Image: The processed output image.
    """
    return RandomResizedCrop((image.size[1], image.size[0]), scale=(0.8, 1.0))(image)


def noise(image: Image) -> Image:
    """Add random variance noise with to test classifier resilience.

       Adapted from:
       https://github.com/RUB-SysSec/GANDCTAnalysis/
       -> create_perturbed_imagedata.py

    Args:
        image (Image): The input PIL.Image .

    Returns:
        Image: Output image with added noise.
    """
    image = np.array(image)
    # variance from U[5.0,20.0]
    variance = np.random.uniform(low=5.0, high=20.0)
    image = np.copy(image).astype(np.float64)
    noise = variance * np.random.randn(*image.shape)
    image += noise
    return Image.fromarray(np.clip(image, 0.0, 255.0).astype(np.uint8))


def blur(image: Image) -> Image:
    """Apply a gaussian blur for resilience testing.

       Adapted from:
       https://github.com/RUB-SysSec/GANDCTAnalysis/
       -> create_perturbed_imagedata.py

    Args:
        image (Image): The PIL.Image input.

    Returns:
        Image: Blurred output.
    """
    # kernel size from [1, 3, 5, 7, 9]
    image = np.array(image)
    kernel_size = np.random.choice([3, 5, 7, 9])
    blurred = cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=cv2.BORDER_DEFAULT
    )
    return Image.fromarray(np.clip(blurred, 0.0, 255.0).astype(np.uint8))
