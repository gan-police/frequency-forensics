""" The original prepare dataset code does not use batch
processing and is, therefore, quite slow. This module
is an attempt to fix this.
"""

import argparse
import functools
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .wavelet_math import batch_packet_preprocessing, identity_processing


def get_label_of_folder(path_of_folder: Path, binary_classification: bool = False) -> int:
    label_str = path_of_folder.name.split("_")[0]
    if binary_classification:
        # differentiate original and generated data
        if label_str == "A":
            return 0
        else:
            return 1
    else:
        # the the label based on the path, As are 0s, Bs are 1, etc.
        if label_str == "A":
            label = 0
        elif label_str == "B":
            label = 1
        elif label_str == "C":
            label = 2
        elif label_str == "D":
            label = 3
        elif label_str == "E":
            label = 4
        else:
            raise NotImplementedError(label_str)
        return label


def get_label(path_to_image: Path, binary_classification: bool) -> int:
    return get_label_of_folder(path_to_image.parent, binary_classification)


def load_and_stack(path_list: list, binary_classification: bool = False) -> tuple:
    image_list = []
    label_list = []
    for path_to_image in path_list:
        image_list.append(np.array(Image.open(path_to_image)))
        label_list.append(np.array(get_label(path_to_image, binary_classification)))
    return np.stack(image_list), label_list


def save_to_disk(
        data_set: np.array, directory: str, previous_file_count: int = 0, dir_suffix: str = ""
) -> int:
    # loop over the batch dimension
    if not os.path.exists(f"{directory}{dir_suffix}"):
        print("creating", f"{directory}{dir_suffix}", flush=True)
        os.mkdir(f"{directory}{dir_suffix}")
    file_count = previous_file_count
    for pre_processed_image in data_set:
        with open(f"{directory}{dir_suffix}/{file_count:06}.npy", "wb") as numpy_file:
            np.save(numpy_file, pre_processed_image)
        file_count += 1

    return file_count


def load_process_store(
        file_list,
        preprocessing_batch_size,
        process, target_dir, label_string, dir_suffix="", binary_classification: bool = False
):
    splits = int(len(file_list) / preprocessing_batch_size)
    batched_files = np.array_split(file_list, splits)
    file_count = 0
    directory = str(target_dir) + "_" + label_string
    all_labels = []
    for current_file_batch in batched_files:
        # load, process and store the current batch training set.
        image_batch, labels = load_and_stack(current_file_batch, binary_classification=binary_classification)
        all_labels.extend(labels)
        processed_batch = process(image_batch)
        file_count = save_to_disk(processed_batch, directory, file_count, dir_suffix)
        print(file_count, label_string, "files processed", flush=True)

    # save labels
    with open(f"{directory}{dir_suffix}/labels.npy", "wb") as label_file:
        np.save(label_file, np.array(all_labels))


def load_folder(folder: Path, train_size: int, val_size: int, test_size: int):
    file_list = list(folder.glob("./*.png"))

    assert (
            len(file_list) >= train_size + val_size + test_size
    ), "Requested set sizes must be smaller or equal to the number of images available."

    # split the list into training, validation and test sub-lists.
    train_list = file_list[:train_size]
    validation_list = file_list[train_size: (train_size + val_size)]
    test_list = file_list[(train_size + val_size): (train_size + val_size + test_size)]

    return np.asarray([train_list, validation_list, test_list], dtype=object)


def pre_process_folder(
        data_folder: str,
        preprocessing_batch_size: int,
        train_size: int,
        val_size: int,
        test_size: int,
        feature: Optional[str] = None,
        missing_label: int = None,
        gan_split_factor: float = 1.0
) -> None:
    """Preprocess a folder containing sub-directories with images from
    different sources. The sub-directories are expected to indicated the
    label in their name. A - for real and B - for GAN generated imagery.

    Args:
        data_folder (str): The folder with the real and gan generated image folders.
        preprocessing_batch_size (int): The batch_size used for image conversion.
        train_size (int): Desired size of the test subset of each folder.
        val_size (int): Desired size of the validation subset of each folder.
        test_size (int): Desired size of the test subset of each folder.
        feature (str): The feature to pre-compute (choose packets or None).
        missing_label (int): label to leave out of training and validation set (choose from {0, 1, 2, 3, 4, None})
        gan_split_factor (float): factor by which the training and validation subset sizes are scaled for each GAN, if
            a missing label is specified.
    """
    data_dir = Path(data_folder)
    target_dir = data_dir.parent / f"{data_dir.name}_{feature}"

    if feature == "packets":
        processing_function = batch_packet_preprocessing
    else:
        processing_function = identity_processing  # type: ignore

    folder_list = sorted(data_dir.glob("./*"))

    if missing_label is not None:
        # split files in folders into training/validation/test
        func_load_folder = functools.partial(load_folder, train_size=train_size, val_size=val_size,
                                             test_size=test_size)

        train_list = []
        validation_list = []
        test_list = []

        for folder in folder_list:
            if get_label_of_folder(folder) == missing_label:
                test_list.extend(load_folder(folder, train_size=0, val_size=0, test_size=test_size)[2])

            else:
                # real data
                if get_label_of_folder(folder, binary_classification=True) == 0:
                    train_result, val_result, test_result = load_folder(folder,
                                                                        train_size=train_size,
                                                                        val_size=val_size,
                                                                        test_size=test_size)
                # generated data
                else:
                    train_result, val_result, test_result = load_folder(folder,
                                                                        train_size=int(train_size * gan_split_factor),
                                                                        val_size=int(val_size * gan_split_factor),
                                                                        test_size=test_size)
                train_list.extend(train_result)
                validation_list.extend(val_result)
                test_list.extend(test_result)

    else:

        # split files in folders into training/validation/test
        func_load_folder = functools.partial(load_folder, train_size=train_size, val_size=val_size, test_size=test_size)
        with ThreadPoolExecutor(max_workers=len(folder_list)) as pool:
            results = list(pool.map(func_load_folder, folder_list))
        results = np.array(results)

        train_list = [img for folder in results[:, 0] for img in folder]
        validation_list = [img for folder in results[:, 1] for img in folder]
        test_list = [img for folder in results[:, 2] for img in folder]

    random.seed(42)
    random.shuffle(train_list)
    random.shuffle(validation_list)
    random.shuffle(test_list)

    if missing_label is not None:
        dir_suffix = f"_missing_{missing_label}"
    else:
        dir_suffix = ""

    binary_classification = missing_label is not None

    # group the sets into smaller batches to go easy on the memory.
    print('processing validation set.', flush=True)
    load_process_store(
        validation_list, preprocessing_batch_size, processing_function, target_dir, "val", dir_suffix=dir_suffix,
        binary_classification=binary_classification
    )
    print("validation set stored")

    # do not use binary label in test set to make performance measurements on the different classes possible
    print("processing test set", flush=True)
    load_process_store(
        test_list, preprocessing_batch_size, processing_function, target_dir, "test", dir_suffix=dir_suffix,
        binary_classification=False
    )
    print("test set stored")

    print("processing training set", flush=True)
    load_process_store(
        train_list, preprocessing_batch_size, processing_function, target_dir, "train", dir_suffix=dir_suffix,
        binary_classification=binary_classification
    )
    print("training set stored.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        type=str,
        help="The folder with the real and gan generated image folders.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=63_000,
        help="Desired size of the training subset of each folder. (default: 63_000).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=5_000,
        help="Desired size of the test subset of each folder. (default: 5_000).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2_000,
        help="Desired size of the validation subset of each folder. (default: 2_000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="The batch_size used for image conversion. (default: 2048).",
    )
    parser.add_argument(
        "--packets",
        "-p",
        help="Save image data as wavelet packets.",
        action="store_true",
    )
    parser.add_argument(
        "--missing-label",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=None,
        help="leave this label out of the training and validation set. Used to test how the models generalize to new "
             "GANs."
    )
    parser.add_argument(
        "--gan-split-factor",
        type=float,
        default=1 / 3,
        help="scaling factor for GAN subsets in the binary classification split. If a missing label is specified, the "
             "classification task changes to classifying whether the data was generated or not. In this case, the share"
             " of the GAN subsets in the split sets should be reduced to balance both classes (i.e. real and generated "
             "data). So, for each GAN the training and validation split subset sizes are then calculated as the general"
             " subset size in the split (i.e. the size specified by '--train-size' etc.) times this factor."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    feature = "packets" if args.packets else "raw"
    pre_process_folder(
        args.directory,
        args.batch_size,
        args.train_size,
        args.val_size,
        args.test_size,
        feature,
        missing_label=args.missing_label,
        gan_split_factor=args.gan_split_factor
    )
    # pre_process_folder('data/source_data/', args.batch_size, args.train_size,
    #                    args.val_size, args.test_size, 'packets')
