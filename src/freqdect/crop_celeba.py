"""Script for cropping celebA adopted from: https://github.com/ningyu1991/GANFingerprints/
This version is taken as is from: https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/crop_celeba.py"""
import argparse
import os
from typing import Tuple

from PIL import Image
import numpy as np

from concurrent.futures import ProcessPoolExecutor


def crop_image(packed: Tuple[int, str, str, str]):
    """Center-crops an CelebA image to 128x128 pixels.

        Args:
            packed (Tuple[int, str, str, str]): Packed args as tuple. The first entry is the image index.
                The second entry is the path of the directory containing all original CelebA images.
                The third entry is the file path of the original image file, which is cropped.
                The fourth entry is the path of the directory where the cropped image is stored.
    """
    i, directory, file_path, output = packed
    if (
        file_path.endswith("png")
        or file_path.endswith("jpeg")
        or file_path.endswith("jpg")
    ):
        image = np.asarray(Image.open(f"{directory}/{file_path}"))

        if image.shape[0] != 128 or image.shape[1] != 128:
            x, y, _ = image.shape
            image = np.copy(image)
            x_upper = min(121 + 64, x)
            y_upper = min(89 + 64, y)
            image = image[x_upper - 128 : x_upper, y_upper - 128 : y_upper]
            image = np.clip(image, 0, 255.0).astype(np.uint8)

        if not (image.shape[0] == 128 and image.shape[1] == 128):
            print("Aborting")
            return i

        Image.fromarray(image).save(f"{output}/celeba_{file_path}")
        return i


def main(args):
    """ Center-crops and resizes a number of CelebA images in a directory to 128x128 pixels and stores the cropped images."""
    os.makedirs(args.OUTPUT, exist_ok=True)
    paths = os.listdir(args.DIRECTORY)[: args.SIZE]
    packed = map(lambda x: (x[0], args.DIRECTORY, x[1], args.OUTPUT), enumerate(paths))

    with ProcessPoolExecutor() as pool:
        jobs = pool.map(crop_image, packed)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Source directory.", type=str)
    parser.add_argument("OUTPUT", help="Output directory.", type=str)
    parser.add_argument("SIZE", help="Amount of data to convert.", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
