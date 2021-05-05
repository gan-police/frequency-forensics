""" Prepare image data.
    Based on
    https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/prepare_dataset.py
"""
import argparse
import functools
import os
from pathlib import Path

import numpy as np

from .image_np import dct2, load_image, normalize, scale_image
from .dct_math import log_scale, welford
from .wavelet_math import packet_preprocessing


# TRAIN_SIZE = 64_000
# VAL_SIZE = 4_000
# TEST_SIZE = 10_000
TRAIN_SIZE = 63_000
VAL_SIZE = 2_000
TEST_SIZE = 5_000


def _find_images(data_path):
    paths = list(Path(data_path).glob("*.jpeg"))

    if len(paths) == 0:
        paths = list(Path(data_path).glob("*.jpg"))
    if len(paths) == 0:
        paths = list(Path(data_path).glob("*.png"))

    return paths


def image_paths(dir_path):
    """Find all filepaths for images in dir_path."""
    return [str(path.absolute()) for path in sorted(_find_images(dir_path))]


def _collect_image_paths(directory):
    images = list(sorted(image_paths(directory)))
    assert len(images) >= TRAIN_SIZE + VAL_SIZE + \
        TEST_SIZE, f"{len(images)} - {directory}"

    train_dataset = images[:TRAIN_SIZE]
    val_dataset = images[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]
    test_dataset = images[TRAIN_SIZE +
                          VAL_SIZE: TRAIN_SIZE + VAL_SIZE + TEST_SIZE]
    assert len(
        train_dataset) == TRAIN_SIZE, f"{len(train_dataset)} - {directory}"

    assert len(val_dataset) == VAL_SIZE, f"{len(val_dataset)} - {directory}"

    assert len(test_dataset) == TEST_SIZE, f"{len(test_dataset)} - {directory}"

    return (train_dataset, val_dataset, test_dataset)


def collect_all_paths(dirs):
    directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(dirs).iterdir())))

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for i, directory in enumerate(directories):
        train, val, test = _collect_image_paths(directory)

        train = zip(train, [i] * len(train))
        val = zip(val, [i] * len(val))
        test = zip(test, [i] * len(test))

        train_dataset.extend(train)
        val_dataset.extend(val)
        test_dataset.extend(test)

        del train, val, test

    train_dataset = np.asarray(train_dataset)
    val_dataset = np.asarray(val_dataset)
    test_dataset = np.asarray(test_dataset)

    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    return train_dataset, val_dataset, test_dataset


def convert_images(inputs, load_function, transformation_function=None, 
                   absolute_function=None, normalize_function=None):
    image, label = inputs
    image = load_function(image)
    if transformation_function is not None:
        image = transformation_function(image)

    if absolute_function is not None:
        image = absolute_function(image)

    if normalize_function is not None:
        image = normalize_function(image)

    return (image, label)


def _dct2_wrapper(image, log=False):
    image = np.asarray(image)
    image = dct2(image)
    if log:
        image = log_scale(image)

    return image


def create_directory_np(output_path, images, convert_function):
    os.makedirs(output_path, exist_ok=True)
    converted_images = map(convert_function, images)

    labels = []
    for i, (img, label) in enumerate(converted_images):
        print(f"\rConverted {i:06d} images!", end="")
        with open(f"{output_path}/{i:06}.npy", "wb+") as f:
            np.save(f, img)

        labels.append(label)

    with open(f"{output_path}/labels.npy", "wb+") as f:
        np.save(f, labels)


def normal_mode(directory, encode_function, outpath):
    (train_dataset, val_dataset, test_dataset) = collect_all_paths(directory)
    create_directory_np(f"{outpath}_train",
                        train_dataset, encode_function)
    print(f"\nConverted train images!")
    create_directory_np(f"{outpath}_val",
                        val_dataset, encode_function)
    print(f"\nConverted val images!")
    create_directory_np(f"{outpath}_test",
                        test_dataset, encode_function)
    print(f"\nConverted test images!")


def main(args):
    output = f"{args.DIRECTORY.rstrip('/')}"

    # we always load images into numpy arrays
    # we additionally set a flag if we later convert to tensorflow records
    load_function = load_image
    transformation_function = None
    normalize_function = None
    absolute_function = None

    if args.color:
        load_function = functools.partial(load_function, grayscale=False)
        output += "_color"

    # dct or raw image data?
    if args.dct:
        output += "_dct"
        transformation_function = _dct2_wrapper

        if args.log:
            # log sclae only for dct coefficients
            assert args.raw is False

            transformation_function = functools.partial(
                _dct2_wrapper, log=True)
            output += "_log_scaled"

        if args.abs:
            # normalize to zero mean and unit variance
            train, _, _ = collect_all_paths(args.DIRECTORY)
            train = train[:TRAIN_SIZE * len(args.DIRECTORY) * 0.1]
            images = map(lambda x: x[0], train)
            images = map(load_function, images)
            images = map(transformation_function, images)

            first = next(images)
            current_max = np.absolute(first)
            for data in images:
                max_values = np.absolute(data)
                mask = current_max > max_values
                current_max *= mask
                current_max += max_values * ~mask

            def scale_by_absolute(image):
                return image / current_max

            absolute_function = scale_by_absolute

        # if args.normalize:
        #    # normalize to zero mean and unit variance
        #    train, _, _ = collect_all_paths(args.DIRECTORY)
        #    images = map(lambda x: x[0], train)
        #    images = map(load_function, images)
        #    images = map(transformation_function, images)
        #    if absolute_function is not None:
        #        images = map(absolute_function, images)

        #    mean, var = welford(images)
        #    std = np.sqrt(var)
        #    output += "_normalized"
        #    normalize_function = functools.partial(
        #        normalize, mean=mean, std=std)

    elif args.packets:
        output += "_packets"
        transformation_function = packet_preprocessing

        # if args.normalize:
        #    # normalize to zero mean and unit variance
        #    train, _, _ = collect_all_paths(args.DIRECTORY)
        #    images = map(lambda x: x[0], train)
        #    images = map(load_function, images)
        #    images = map(transformation_function, images)
        #    if absolute_function is not None:
        #        images = map(absolute_function, images)

        #    mean, var = welford(images)
        #    std = np.sqrt(var)
        #    output += "_normalized"
        #    normalize_function = functools.partial(
        #        normalize, mean=mean, std=std)

    else:
        output += "_raw"

        # normalization sclaes to [-1, 1]
        if args.normalize:
            normalize_function = scale_image
            output += "_normalized"

    encode_function = functools.partial(convert_images, load_function=load_function,
                                        transformation_function=transformation_function,
                                        normalize_function=normalize_function, absolute_function=absolute_function)
    normal_mode(args.DIRECTORY, encode_function, output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", type=str,
                        help="Directory to convert.")
    parser.add_argument("--dct", "-r", help="Save image data as dct image.",
                        action="store_true")
    parser.add_argument("--packets", "-p", help="Save image data as dct image.",
                        action="store_true")
    parser.add_argument("--log", "-l", help="Log scale Images.",
                        action="store_true")
    parser.add_argument("--abs", "-a", help="Scale each feature by its max absolute value.",
                        action="store_true")
    parser.add_argument("--color", "-c", help="Compute as color instead.",
                        action="store_true")
    parser.add_argument("--normalize", "-n", help="Normalize data.",
                        action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
