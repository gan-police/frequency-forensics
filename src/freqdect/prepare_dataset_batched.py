""" The original prepare dataset code does not use batch
processing and is, therefore, quite slow. This module
is an attempt to fix this.
"""
import os
import argparse
from pathlib import Path
import random
import numpy as np
from PIL import Image
from .wavelet_math import batch_packet_preprocessing, identity_processing


def get_label(path_to_image: Path) -> int:
    # the the label based on the path, As are 0s and Bs are 1.
    label_str = path_to_image.parent.name.split("_")[0]
    if label_str == 'A':
        label = 0
    elif label_str == 'B':
        label = 1
    elif label_str == 'C':
        label = 2
    elif label_str == 'D':
        label = 3
    else:
        raise NotImplementedError(label_str)
    return label


def load_and_stack(path_list: list) -> (np.array, np.array):
    image_list = []
    label_list = []
    for path_to_image in path_list:
        image_list.append(np.array(Image.open(path_to_image)))
        label_list.append(np.array(get_label(path_to_image)))
    return np.stack(image_list), np.stack(label_list)


def preprocess(image_batch_list: list, process: callable) -> np.array:
    preprocessed_images_list = []
    for image_batch in image_batch_list:
        preprocessed_images_list.append(process(image_batch))
    return np.concatenate(preprocessed_images_list, axis=0)


def save_to_disk(data_set: np.array, labels: np.array, directory: str) -> None:
    # loop over the batch dimension
    os.mkdir(directory)
    for number, pre_processed_image in enumerate(data_set):
        with open(f"{directory}/{number:06}.npy", "wb") as numpy_file:
            np.save(numpy_file, pre_processed_image)

    with open(f"{directory}/labels.npy", "wb") as label_file:
        np.save(label_file, labels)


def pre_process_folder(data_folder: str, preprocessing_batch_size: int, train_size: int,
                       val_size: int, test_size: int, feature: str = None) -> None:
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
    target_dir = data_dir.parent / (data_dir.name + "_" + feature)

    if feature == 'packets':
        processing_function = batch_packet_preprocessing
    else:
        processing_function = identity_processing

    # find all files in the data_folders
    folder_list = data_dir.glob('./*')
    file_list = []
    for folder in folder_list:
        files = folder.glob('./*.png')
        file_list.extend(files)

    # shuffle the list and split it into training, validation and test
    # sub-lists.
    assert len(file_list) >= train_size + val_size + test_size, \
        "Requested set sizes must be smaller or equal to the number of\
         images available."
    random.seed(42)
    random.shuffle(file_list)
    train_list = file_list[:train_size]
    validation_list = file_list[train_size:(train_size + val_size)]
    test_list = file_list[(train_size + val_size):(train_size + val_size + test_size)]

    # load, process and store the training set.
    train_set, train_labels = load_and_stack(train_list)
    splits = int(train_set.shape[0] / preprocessing_batch_size)
    training_preprocessing_batches = np.array_split(train_set, splits)
    del train_set
    processed_train_set = preprocess(training_preprocessing_batches, processing_function)
    save_to_disk(processed_train_set, train_labels, target_dir.parent / (target_dir.name + '_train'))
    del processed_train_set, training_preprocessing_batches
    print('training set stored.')

    validation_set, validation_labels = load_and_stack(validation_list)
    splits = int(validation_set.shape[0] / preprocessing_batch_size)
    validation_preprocessing_batches = np.array_split(validation_set, splits)
    del validation_set
    preprocessed_validation_set = preprocess(validation_preprocessing_batches, processing_function)
    save_to_disk(preprocessed_validation_set, validation_labels, target_dir.parent / (target_dir.name + '_val'))
    del preprocessed_validation_set, validation_preprocessing_batches
    print('validation set stored')

    test_set, test_labels = load_and_stack(test_list)
    splits = int(test_set.shape[0] / preprocessing_batch_size)
    test_preprocessing_batches = np.array_split(test_set, splits)
    del test_set
    preprocessed_test_set = preprocess(test_preprocessing_batches, processing_function)
    save_to_disk(preprocessed_test_set, test_labels, target_dir.parent / (target_dir.name + '_test'))
    del preprocessed_test_set, test_preprocessing_batches
    print('test set stored')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("directory", type=str,
                        help="The folder with the real and gan generated image folders.")
    parser.add_argument("--train-size", type=int, default=2*63_000,
                        help="Desired size of the training set. (default: 126_000).")
    parser.add_argument("--test-size", type=int, default=2 * 5_000,
                        help="Desired size of the test set. (default: 5_000).")
    parser.add_argument("--val-size", type=int, default=2 * 2_000,
                        help="Desired size of the validation set. (default: 4_000).")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="The batch_size used for image conversion. (default: 2048).")
    parser.add_argument("--packets", "-p", help="Save image data as wavelet packets.", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    feature = 'packets' if args.packets else 'raw'
    pre_process_folder(args.directory, args.batch_size, args.train_size, args.val_size, args.test_size, feature)
