"""Test the corruption code used for robustness testing."""
import sys

import numpy as np
from PIL import Image
from scipy import misc

sys.path.append("./src")

from freqdect.corruption import jpeg_compression, random_resized_crop, random_rotation


def test_jpeg_compression():
    """Test the jepeg compression function."""
    face = Image.fromarray(misc.face())
    compressed = np.array(jpeg_compression(face))
    assert np.array(face).shape == compressed.shape  # noqa: S101


def test_rotation():
    """Test the random rotation function from freqdect.corruption."""
    face = Image.fromarray(misc.face())
    rotated = random_rotation(face)
    assert face.size == rotated.size  # noqa: S101


def test_crop():
    """Test the random cropping function from freqdect.corruption."""
    face = Image.fromarray(misc.face())
    crop = random_resized_crop(face)
    assert crop.size == face.size  # noqa: S101


if __name__ == "__main__":
    test_jpeg_compression()
    test_rotation()
    test_crop()

    import matplotlib.pyplot as plt

    face = Image.fromarray(misc.face())
    plt.imshow(np.array(face))
    plt.show()

    second_face = np.array(random_rotation(jpeg_compression(face)))

    plt.imshow(second_face)
    plt.show()

    third_face = random_resized_crop(face)
    plt.imshow(third_face)
    plt.show()
