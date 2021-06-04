""" Speedy preprocessing using batched PyTorch code.
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


def get_label(path_to_image: Path) -> int:
    """Get the label based on the image path.
        We assume:
            A: Orignal data, B: First gan,
            C: Second gan, D: Third gan, E: Fourth gan.
        A working folder structure could look like:
            A_celeba  B_CramerGAN  C_MMDGAN  D_ProGAN  E_SNGAN
        With each folder containing the images from the corresponding
        source.

    Args:
        path_to_image (Path): Image path string containing only a single
            underscore direcrly after the label letter.

    Raises:
        NotImplementedError: Raised if the label letter is unkown.

    Returns:
        int: The label encoded as integer.
    """
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
    elif label_str == "E":
        label = 4
    else:
        raise NotImplementedError(label_str)
    return label


def load_and_stack(path_list: list) -> tuple:
    """Transform a lists of paths into a batches of
    numpy arrays and record their labels.

    Args:
        path_list (list): A list of Poxis paths strings.
            The stings must follow the convention outlined
            in the get_label function.

    Returns:
        tuple: A numpy array of size
            (preprocessing_batch_size, height, width, 3)
            and a list of length preprocessing_batch_size.
    """
    image_list = []
    label_list = []
    for path_to_image in path_list:
        image_list.append(np.array(Image.open(path_to_image)))
        label_list.append(np.array(get_label(path_to_image)))
    return np.stack(image_list), label_list


def save_to_disk(
    data_batch: np.array, directory: str, previous_file_count: int = 0
) -> int:
    """Save images to disk using their position on the dataset as filename.

    Args:
        data_batch (np.array): The image batch to store.
        directory (str): The place to store the images at.
        previous_file_count (int, optional): The number of previously stored images.
            Defaults to 0.

    Returns:
        int: The new total of storage images.
    """
    # loop over the batch dimension
    if not os.path.exists(directory):
        print("creating", directory, flush=True)
        os.mkdir(directory)
    file_count = previous_file_count
    for pre_processed_image in data_batch:
        with open(f"{directory}/{file_count:06}.npy", "wb") as numpy_file:
            np.save(numpy_file, pre_processed_image)
        file_count += 1

    return file_count


def load_process_store(
    file_list, preprocessing_batch_size, process, target_dir, label_string
):
    """Loads processes and stores a file list according to a processing function.

    Args:
        file_list (list): PosixPath objects leading to source images.
        preprocessing_batch_size (int): The number of files processed at once.
        process (function): The function to use for the preprocessing.
            I.e. a wavelet packet encoding.
        target_dir (string): A directory where to save the processed files.
        label_string (string): A label we add to the target folder.
    """
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
        print(file_count, label_string, "files processed", flush=True)

    # save labels
    with open(f"{directory}/labels.npy", "wb") as label_file:
        np.save(label_file, np.array(all_labels))


def load_folder(
    folder: Path, train_size: int, val_size: int, test_size: int
) -> np.array:
    """Given a folder containing portable network graphics (*.png) files
       this functions will create Posix-path lists. A train, test, and
       validation set list is created.

    Args:
        folder (Path): Path to a folder with images from the same source.
            I.e. A_ffhq .
        train_size (int): Desired size of the training set.
        val_size (int): Desired size of the validation set.
        test_size (int): Desired size of the test set.

    Returns:
        np.array: Numpy array with the train, validation and test lists,
            in this order.
    """
    file_list = list(folder.glob("./*.png"))

    assert (
        len(file_list) >= train_size + val_size + test_size
    ), "Requested set sizes must be smaller or equal to the number of images available."

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
) -> None:
    """Preprocess a folder containing sub-directories with images from
    different sources. All images are expected to have the same size.
    The sub-directories are expected to indicate to label their source in
    their name. For example,  A - for real and B - for GAN generated imagery.

    Args:
        data_folder (str): The folder with the real and gan generated image folders.
        preprocessing_batch_size (int): The batch_size used for image conversion.
        train_size (int): Desired size of the test subset of each folder.
        val_size (int): Desired size of the validation subset of each folder.
        test_size (int): Desired size of the test subset of each folder.
        feature (str): The feature to pre-compute (choose packets, log_packets or None).
    """
    data_dir = Path(data_folder)
    target_dir = data_dir.parent / f"{data_dir.name}_{feature}"

    if feature == "packets":
        processing_function = batch_packet_preprocessing
    if feature == "log_packets":
        processing_function = functools.partial(
            batch_packet_preprocessing, log_scale=True
        )
    else:
        processing_function = identity_processing  # type: ignore

    folder_list = sorted(data_dir.glob("./*"))

    # split files in folders into training/validation/test
    func_load_folder = functools.partial(
        load_folder, train_size=train_size, val_size=val_size, test_size=test_size
    )
    with ThreadPoolExecutor(max_workers=len(folder_list)) as pool:
        results = list(pool.map(func_load_folder, folder_list))
    results = np.array(results)

    train_list = [img for folder in results[:, 0] for img in folder]
    validation_list = [img for folder in results[:, 1] for img in folder]
    test_list = [img for folder in results[:, 2] for img in folder]

    # fix the seed to make results reproducible.
    random.seed(42)
    random.shuffle(train_list)
    random.shuffle(validation_list)
    random.shuffle(test_list)

    # group the sets into smaller batches to go easy on the memory.
    print("processing validation set.", flush=True)
    load_process_store(
        validation_list,
        preprocessing_batch_size,
        processing_function,
        target_dir,
        "val",
    )
    print("validation set stored")

    print("processing test set", flush=True)
    load_process_store(
        test_list, preprocessing_batch_size, processing_function, target_dir, "test"
    )
    print("test set stored")

    print("processing training set", flush=True)
    load_process_store(
        train_list, preprocessing_batch_size, processing_function, target_dir, "train"
    )
    print("training set stored.", flush=True)


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
        "--log-packets",
        "-lp",
        help="Save image data as log-scaled wavelet packets.",
        action="store_true",
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
    )