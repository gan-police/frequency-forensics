""" Calculating confusion matrices from trained models
    that classify deepfake image data."""
from PIL import Image
from io import BytesIO
from torchvision.transforms import RandomRotation, RandomResizedCrop


def jpeg_compression(image: Image, jpeg_compression: int) -> Image:
    """ Compute a compressed version of the input image.

    Args:
        image (Image): The input image.
        jpeg_compression (int): Compression factor
          on a scale from 0 (worst) to 95 (best).

    Returns:
        Image: The compressed image.
   """
    out = BytesIO()
    image.save(out, format='JPEG', quality=jpeg_compression)
    return Image.open(out)


def random_rotation(image: Image, angle=25) -> Image:
    return RandomRotation(angle)(image)


def random_resized_crop(image: Image) -> Image:
    return RandomResizedCrop((image.size[1], image.size[0]),
                             scale=(0.8, 1.))(image)
