""" The original prepare dataset code does not use batch
processing and is, therefore, quite slow. This module
is an attempt to fix this.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .wavelet_math import batch_packet_preprocessing, identity_processing


def get_label(path_to_image: Path) -> int:
    # the the label based on the path, As are 0s and Bs are 1.
    label_str = path_to_image.parent.name.split("_")[0]
    if label_str == "A":
        label = 0
    elif label_str == "B":
        label = 1
    elif label_str == "C":
        label = 2
    elif label_str == "D":
        label = 3
    else:
        raise NotImplementedError(label_str)
    return label


def load_and_stack(path_list: list) -> tuple:
    image_list = []
    label_list = []
    for path_to_image in path_list:
        image_list.append(np.array(Image.open(path_to_image)))
        label_list.append(np.array(get_label(path_to_image)))
    return np.stack(image_list), label_list


def save_to_disk(
    data_set: np.array, directory: str, previous_file_count: int = 0
) -> int:
    # loop over the batch dimension
    if not os.path.exists(directory):
        print("creating", directory)
        os.mkdir(directory)
    file_count = previous_file_count
    for pre_processed_image in data_set:
        with open(f"{directory}/{file_count:06}.npy", "wb") as numpy_file:
            np.save(numpy_file, pre_processed_image)
        file_count += 1

    return file_count


def load_process_store(
    file_list, preprocessing_batch_size, process, target_dir, label_string
):
    splits = int(len(file_list) / preprocessing_batch_size)
    batched_files = np.array_split(file_list, splits)
    file_count = 0
    directory = str(target_dir) + "_" + label_string
    all_labels = []
    for current_file_batch in batched_files:
        # load, process and store the current batch training set.
        image_batch, labels = load_and_stack(current_file_batch)
        all_labels.extend(labels)
        processed_batch = process(image_batch)
        file_count = save_to_disk(processed_batch, directory, file_count)
        print(file_count, label_string, "files processed")

    # save labels
    with open(f"{directory}/labels.npy", "wb") as label_file:
        np.save(label_file, np.array(all_labels))


def pre_process_folder(
    data_folder: str,
    preprocessing_batch_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    feature: Optional[str] = None,
) -> None:
    """Preprocess a folder containing sub-directories with images from
    different sources. The sub-directories are expected to indicated the
    label in their name. A - for real and B - for GAN generated imagery.

    Args:
        data_folder (str): The folder with the real and gan generated image folders.
        preprocessing_batch_size (int): The batch_size used for image conversion.
        train_size (int): Desired size of the test set.
        val_size (int): Desired size of the validation set.
        test_size (int): Desired size of the test set.
        feature (str): The feature to pre-compute (choose packets or None).
    """
    data_dir = Path(data_folder)
    target_dir = data_dir.parent / f"{data_dir.name}_{feature}"

    if feature == "packets":
        processing_function = batch_packet_preprocessing
    else:
        processing_function = identity_processing  # type: ignore

    random.seed(42)

    # split the files in all data_folders into training/validation/testing
    train_list = []
    validation_list = []
    test_list = []

    # find all files in the data_folders
    folder_list = sorted(data_dir.glob("./*"))
    for folder in folder_list:
        file_list = list(folder.glob("./*.png"))

        assert (
            len(file_list) >= train_size + val_size + test_size
        ), "Requested set sizes must be smaller or equal to the number of images available."

        # shuffle the list and split it into training, validation and test sub-lists.
        random.shuffle(file_list)
        train_list.extend(file_list[:train_size])
        validation_list.extend(file_list[train_size: (train_size + val_size)])
        test_list.extend(file_list[(train_size + val_size): (train_size + val_size + test_size)])

    # group the train set into smaller batches to go easy on the memory.
    print("processing training set")
    load_process_store(
        train_list, preprocessing_batch_size, processing_function, target_dir, "train"
    )
    print("training set stored.")

    load_process_store(
        validation_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "val",
    )
    print("validation set stored")

    load_process_store(
        test_list, preprocessing_batch_size, processing_function, target_dir, "test"
    )

    print("test set stored")


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
        default=2 * 63_000,
        help="Desired size of the training set. (default: 126_000).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2 * 5_000,
        help="Desired size of the test set. (default: 10_000).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2 * 2_000,
        help="Desired size of the validation set. (default: 4_000).",
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
    )
    # pre_process_folder('data/source_data/', args.batch_size, args.train_size,
    #                    args.val_size, args.test_size, 'packets')
