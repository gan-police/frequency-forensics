import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple
from functools import partial

import numpy as np
from PIL import Image


def resize_image(packed: Tuple[int, str, str, str], shape: Tuple[int, int]):
    """Resize an image.

    Args:
        packed (Tuple[int, str, str, str]): Packed args as tuple.
            The first entry is the image index.
            The second entry is the path of the directory containing all original CelebA images.
            The third entry is the file path of the original image file, which is cropped.
            The fourth entry is the path of the directory where the cropped image is stored.
        shape (Tuple(int, int)): The shape for the resized images.
    """
    i, directory, file_path, output = packed
    if (
        file_path.endswith("png")
        or file_path.endswith("jpeg")
        or file_path.endswith("jpg")
    ):
        image = Image.open(f"{directory}/{file_path}")
        image = image.resize(shape)
        image.save(f"{output}/resize_{file_path}")
        return i


def main(args):
    """Resizes a number of images in a directory and stores the resized images."""
    os.makedirs(args.OUTPUT, exist_ok=True)
    paths = os.listdir(args.DIRECTORY)[: args.SIZE]
    packed = map(lambda x: (x[0], args.DIRECTORY, x[1], args.OUTPUT), enumerate(paths))
    packed_list = list(packed)
    print('image total', len(packed_list))
    resize_shape = partial(resize_image, shape=(args.SHAPE, args.SHAPE))

    with ProcessPoolExecutor() as pool:
        _ = pool.map(resize_shape, packed_list)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Source directory.", type=str)
    parser.add_argument("OUTPUT", help="Output directory.", type=str)
    parser.add_argument("SHAPE", help="Shape for the new images.", type=int)
    parser.add_argument("SIZE", help="Amount of data to convert.", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
