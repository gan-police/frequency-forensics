"""Image corruption code for robustness testing."""
from io import BytesIO

from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomRotation


def jpeg_compression(image: Image, jpeg_compression: int) -> Image:
    """Compute a compressed version of the input image.

    Args:
        image (Image): The input image.
        jpeg_compression (int): Compression factor
          on a scale from 0 (worst) to 95 (best).

    Returns:
        Image: The compressed image.
    """
    out = BytesIO()
    image.save(out, format="JPEG", quality=jpeg_compression)
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
    return RandomResizedCrop((image.size[1], image.size[0]), scale=(0.9, 1.0))(image)
