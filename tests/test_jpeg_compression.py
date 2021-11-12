import numpy as np
from PIL import Image
from scipy import misc
import sys

sys.path.append("./src")
from freqdect.pre_processing import jpeg_compression, random_rotation, random_resized_crop


def test_jpeg_compression():
    face = Image.fromarray(misc.face())
    compressed = np.array(jpeg_compression(face, 95))
    err = np.mean(np.abs(np.array(face) - compressed))
    assert err < 40
    assert np.array(face).shape == compressed.shape


def test_rotation():
    face = Image.fromarray(misc.face())
    rotated = random_rotation(face)
    assert face.size == rotated.size


def test_crop():
    face = Image.fromarray(misc.face())
    crop = random_resized_crop(face)
    assert crop.size == face.size


if __name__ == '__main__':
    test_jpeg_compression()
    test_rotation()
    test_crop()

    import matplotlib.pyplot as plt
    face = Image.fromarray(misc.face())
    plt.imshow(np.array(face))
    plt.show()

    second_face = np.array(random_rotation(jpeg_compression(face, 100)))

    plt.imshow(second_face)
    plt.show()

    third_face = random_resized_crop(face)
    plt.imshow(third_face)
    plt.show()
