"""Preprocessing for the StarGAN experiment based on prepare_dataset module."""

import argparse
import functools
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .prepare_dataset import load_process_store
from .data_loader import LoadNumpyDataset
from .wavelet_math import batch_packet_preprocessing, identity_processing


def load_folder(
    folder: Path, train_size: int, val_size: int, test_size: int
) -> np.array:
    """Create posix-path lists for png files in a folder.

    Given a folder containing *.jpg files this functions will create Posix-path lists.
    A train, test, and validation set list is created.

    Args:
        folder: Path to a folder with images from the same source, i.e. A_ffhq .
        train_size: Desired size of the training set.
        val_size: Desired size of the validation set.
        test_size: Desired size of the test set.

    Returns:
        Numpy array with the train, validation and test lists, in this order.

    Raises:
        ValueError: if the requested set sizes are not smaller or equal to the number of images available

    # noqa: DAR401
    """
    file_list = list(folder.glob("./*.jpg"))
    random.shuffle(file_list)
    if len(file_list) < train_size + val_size + test_size:
        raise ValueError(
            "Requested set sizes must be smaller or equal to the number of images available."
        )

    # split the list into training, validation and test sub-lists.
    train_list = file_list[:train_size]
    validation_list = file_list[train_size : (train_size + val_size)]
    test_list = file_list[(train_size + val_size) : (train_size + val_size + test_size)]

    return np.asarray([train_list, validation_list, test_list], dtype=object)


def pre_process_folder(
    data_folder: str,
    preprocessing_batch_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    feature: Optional[str] = None,
    wavelet: str = "db1",
    boundary: str = "reflect",
    jpeg_compression_number: int = None,
    crop_rotate: bool = False,
) -> None:
    """Preprocess a folder containing sub-directories with images from different sources.

    All images are expected to have the same size.
    The sub-directories are expected to indicate to label their source in
    their name. For example,  A - for real and B - for GAN generated imagery.

    Args:
        data_folder (str): The folder with the real and gan generated image folders.
        preprocessing_batch_size (int): The batch_size used for image conversion.
        train_size (int): Desired size of the test subset of each folder.
        val_size (int): Desired size of the validation subset of each folder.
        test_size (int): Desired size of the test subset of each folder.
        feature (str): The feature to pre-compute (choose packets, log_packets or raw).
        jpeg_compression_number (int): jpeg comression factor used for robustness testing.
            Defaults to None.
        rotation_and_crop (bool): If true some images are randomly cropped or rotated.
            Defaults to False.
    """
    # fix the seed to make results reproducible.
    random.seed(42)

    data_dir = Path(data_folder)
    if feature == "raw":
        target_dir = (
            data_dir.parent
            / f"{data_dir.name}_{feature}_j_{jpeg_compression_number}_cr_{crop_rotate}"
        )
    else:
        target_dir = (
            data_dir.parent
            / f"{data_dir.name}_{feature}_{wavelet}_{boundary}_j_{jpeg_compression_number}_cr_{crop_rotate}"
        )

    if feature == "packets":
        processing_function = functools.partial(
            batch_packet_preprocessing, wavelet=wavelet, mode=boundary
        )
    elif feature == "log_packets":
        processing_function = functools.partial(
            batch_packet_preprocessing, log_scale=True, wavelet=wavelet, mode=boundary
        )
    else:
        processing_function = identity_processing  # type: ignore

    folder_list = sorted(data_dir.glob("./*"))

    # Split only original images first and then replace half of them with corresponding StarGAN fakes
    train_list, validation_list, test_list = load_folder(
        folder_list[0], train_size=train_size, val_size=val_size, test_size=test_size
    )

    num_of_classes = len(folder_list) - 1

    def _insert_cls_files(counter, folder, size, lst):
        for idx in range(
            int(counter * 0.5 * size / num_of_classes),
            int((counter + 1) * 0.5 * size / num_of_classes),
        ):
            lst[idx] = folder / lst[idx].name

    for cls_idx, cls_folder in enumerate(folder_list[1:]):
        _insert_cls_files(cls_idx, cls_folder, train_size, train_list)
        _insert_cls_files(cls_idx, cls_folder, val_size, validation_list)
        _insert_cls_files(cls_idx, cls_folder, test_size, test_list)

    # fix the seed to make results reproducible.
    random.shuffle(train_list)
    random.shuffle(validation_list)
    random.shuffle(test_list)

    dir_suffix = ""
    binary_classification = True

    # group the sets into smaller batches to go easy on the memory.
    print("processing validation set.", flush=True)
    load_process_store(
        validation_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "val",
        dir_suffix=dir_suffix,
        binary_classification=binary_classification,
        jpeg_compression_number=jpeg_compression_number,
        rotation_and_crop=crop_rotate,
    )
    print("validation set stored")

    # do not use binary label in test set to make performance measurements on the different classes possible
    print("processing test set", flush=True)
    load_process_store(
        test_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "test",
        dir_suffix=dir_suffix,
        binary_classification=False,
        jpeg_compression_number=jpeg_compression_number,
        rotation_and_crop=crop_rotate,
    )
    print("test set stored")

    print("processing training set", flush=True)
    load_process_store(
        train_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "train",
        dir_suffix=dir_suffix,
        binary_classification=binary_classification,
        jpeg_compression_number=jpeg_compression_number,
        rotation_and_crop=crop_rotate,
    )
    print("training set stored.", flush=True)

    # compute training normalization.
    # load train data and compute mean and std
    print("computing mean and std values.")
    train_data_set = LoadNumpyDataset(f"{target_dir}_train{dir_suffix}")
    img_lst = []
    for img_no in range(train_data_set.__len__()):
        img_lst.append(train_data_set.__getitem__(img_no)["image"])
    img_data = torch.stack(img_lst, 0)
    # average all axis except the color channel
    axis = tuple(np.arange(len(img_data.shape[:-1])))
    # calculate mean and std in double to avoid precision problems
    mean = torch.mean(img_data.double(), axis).float()
    std = torch.std(img_data.double(), axis).float()
    del img_data
    print("mean", mean, "std:", std)
    with open(f"{target_dir}_train{dir_suffix}/mean_std.pkl", "wb") as f:
        pickle.dump([mean.numpy(), std.numpy()], f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        type=str,
        help="The folder with the real and gan generated image folders.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10,
        help="Desired size of the training subset of each folder. (default: 63_000).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="Desired size of the test subset of each folder. (default: 5_000).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=0,
        help="Desired size of the validation subset of each folder. (default: 2_000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="The batch_size used for image conversion. (default: 2048).",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--raw",
        "-r",
        help="Save image data as raw image data.",
        action="store_true",
    )
    group.add_argument(
        "--packets",
        "-p",
        help="Save image data as wavelet packets.",
        action="store_true",
    )
    group.add_argument(
        "--log-packets",
        "-lp",
        help="Save image data as log-scaled wavelet packets.",
        action="store_true",
    )

    parser.add_argument(
        "--wavelet",
        type=str,
        default="haar",
        help="The wavelet to use. Choose one from pywt.wavelist(). Defaults to haar.",
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="reflect",
        help="The boundary treatment method to use. Choose zero, reflect, or boundary. Defaults to reflect.",
    )

    parser.add_argument(
        "--jpeg",
        type=int,
        default=None,
        help="Use jpeg compression to measure the robustness of our method. The compression factor"
        "should be an integer on a scale from 0 (worst) to 95 (best).",
    )

    parser.add_argument(
        "--crop-rotate",
        "-cr",
        action="store_true",
        help="If set some images will be randomly cropped or rotated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.packets:
        feature = "packets"
    elif args.log_packets:
        feature = "log_packets"
    else:
        feature = "raw"

    pre_process_folder(
        args.directory,
        args.batch_size,
        args.train_size,
        args.val_size,
        args.test_size,
        feature,
        wavelet=args.wavelet,
        boundary=args.boundary,
        jpeg_compression_number=args.jpeg,
        crop_rotate=args.crop_rotate,
    )
