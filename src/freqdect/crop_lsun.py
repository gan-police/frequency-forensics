"""Script for cropping LSUN adopted from: https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/crop_lsun.py
which is based on: https://github.com/ningyu1991/GANFingerprints/"""

import argparse
import os

from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def transform_image(stupid):
    file_path, directory, output = stupid
    # catch errors and continue with different files
    try:
        if file_path.endswith("png") or file_path.endswith("jpeg") or file_path.endswith("jpg") or file_path.endswith("webp"):
            image = Image.open(f"{directory}/{file_path}")
            x, y = image.size
            if y < x:
                crop_height = y
                crop_width = y

                crop_left = (x - y)//2
                crop_top = 0
                image = image.crop((crop_left, crop_top, crop_left+crop_width, crop_top+crop_height))
            elif x < y:
                crop_height = x
                crop_width = x

                crop_left = 0
                crop_top = (y - x)//2
                image = image.crop((crop_left, crop_top, crop_left+crop_width, crop_top+crop_height))

            image = image.resize((128, 128))

            # store .webp images as .png files
            if file_path.endswith("webp"):
                file_path = file_path.replace(".webp", ".png")
            image.save(f"{output}/{file_path}")
        else:
            print(f"Skipped {file_path}")
    except ValueError as exc:
        print(file_path, exc, x, y)

def main(args):
    os.makedirs(args.OUTPUT, exist_ok=True)

    # only consider the specified number of files
    paths = os.listdir(args.DIRECTORY)[:args.SIZE]
    packed = map(lambda p: (p, args.DIRECTORY, args.OUTPUT), paths)
    with ProcessPoolExecutor() as pool:
        list(pool.map(transform_image, packed))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Source directory.", type=str)
    parser.add_argument("OUTPUT", help="Output directory.", type=str)

    # added argument "SIZE"
    parser.add_argument("SIZE", help="Amount of data to convert.", type=int)

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
