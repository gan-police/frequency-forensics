"""Sensitivity analysis results processing module."""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from .plot_mean_packets import generate_frequency_packet_image


def process_results_dir_avg(
    data_dir: str,
    img_shape: Tuple[int, int] = (128, 128),
    degree: int = 3,
) -> np.ndarray:
    """Compute average gradients of most probable class.

    Args:
        data_dir (str): Directory in which the results from saliency.py are stored.
        img_shape (Tuple[int, int]): Input (image) shapes. Defaults to (128, 128).
        degree (int): wavelet degree (required to compose single image from wavelets). Defaults to 3.

    Returns:
        np.ndarray : The resulting gradient image [img_shape[0], img_shape[1]]
    """

    def process_raw_image(s, o):
        # s.shape = (classes, H, W, C)
        p = np.exp(o)  # p(class)
        i = np.argmax(p)
        x = s[i]  # take more probable class
        x = np.abs(np.mean(x, axis=-1))  # abs mean over channel
        x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x  # [0,1]
        return x.astype(float)

    def process_wavelet_image(s, o):
        # s.shape = (classes, wavelets, H, W, C)
        p = np.exp(o)  # p(class)
        i = np.argmax(p)
        x = s[i]  # take more probable class
        x = np.abs(np.mean(x, axis=-1))  # abs mean over channel
        x = generate_frequency_packet_image(x, degree)  # compose
        x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x  # [0,1]
        return x.astype(float)

    def process_image(s, o):
        if len(s.shape) == 4:
            return process_raw_image(s, o)
        elif len(s.shape) == 5:
            return process_wavelet_image(s, o)
        else:
            raise NotImplementedError(f"Data shape {s.shape} cannot be processed")

    def process_batch(s_in, o_in):
        batch_picture = np.zeros(img_shape, dtype=float)
        for s, o in zip(s_in, o_in):
            batch_picture += process_image(s, o)
        return batch_picture, s_in.shape[0]

    print("avg")
    avg_picture = np.zeros(img_shape, dtype=float)
    counter = 0
    file_lst = sorted(Path(data_dir).glob("./*.npy"))
    for file in tqdm(file_lst, desc="process files"):
        data = np.load(file)
        batch_picture, batch_counter = process_batch(data["S"], data["O"])
        avg_picture += batch_picture
        counter += batch_counter

    print(f"\tprocessed {counter} images")
    return avg_picture / counter


def process_results_dir_std(
    data_dir: str,
    avg_picture: np.ndarray,
    img_shape: Tuple[int, int] = (128, 128),
    degree: int = 3,
) -> np.ndarray:
    """Compute standard deviation of gradients of most probable class.

    Args:
        data_dir (str): Directory in which the results from saliency.py are stored.
        avg_picture (np.ndarray): Average gradient from process_results_dir_avg.
        img_shape (Tuple[int, int]): Input (image) shapes. Defaults to (128, 128).
        degree (int): wavelet degree (required to compose single image from wavelets). Defaults to 3.

    Returns:
        np.ndarray : The resulting gradient image [img_shape[0], img_shape[1]]
    """

    def process_raw_image(s, o):
        # s.shape = (classes, H, W, C)
        p = np.exp(o)  # p(class)
        i = np.argmax(p)
        x = s[i]  # take more probable class
        x = np.abs(np.mean(x, axis=-1))  # abs mean over channel
        x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x  # [0,1]
        return x.astype(float)

    def process_wavelet_image(s, o):
        # s.shape = (classes, wavelets, H, W, C)
        p = np.exp(o)  # p(class)
        i = np.argmax(p)
        x = s[i]  # take more probable class
        x = np.abs(np.mean(x, axis=-1))  # abs mean over channel
        x = generate_frequency_packet_image(x, degree)  # compose
        x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x  # [0,1]
        return x.astype(float)

    def process_image(s, o):
        if len(s.shape) == 4:
            return process_raw_image(s, o)
        elif len(s.shape) == 5:
            return process_wavelet_image(s, o)
        else:
            raise NotImplementedError(f"Data shape {s.shape} cannot be processed")

    def process_batch(s_in, o_in):
        batch_picture = np.zeros(img_shape, dtype=float)
        for s, o in zip(s_in, o_in):
            batch_picture += (process_image(s, o) - avg_picture) ** 2
        return batch_picture, s_in.shape[0]

    print("std")
    std_picture = np.zeros(img_shape, dtype=float)
    counter = 0
    file_lst = sorted(Path(data_dir).glob("./*.npy"))
    for file in tqdm(file_lst, desc="process files"):
        data = np.load(file)
        batch_picture, batch_counter = process_batch(data["S"], data["O"])
        std_picture += batch_picture
        counter += batch_counter

    print(f"\tprocessed {counter} images")
    return np.sqrt(std_picture / counter)


def main(args):
    """Process results from saliency.py ."""
    print(f"Process '{args.sal_dir}'")

    avg_picture = process_results_dir_avg(args.sal_dir)
    std_picture = process_results_dir_std(args.sal_dir, avg_picture)

    array_dict = {"avg": avg_picture, "std": std_picture}
    if not os.path.exists(args.result_dir):
        print("creating", args.result_dir)
        os.mkdir(args.result_dir)
    prefix = os.path.split(args.sal_dir)[-1]
    filename = f"{prefix}-result.npy"
    with open(os.path.join(args.result_dir, filename), "wb") as numpy_file:
        np.savez(numpy_file, **array_dict)

    print(f"Finished: {filename}.")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sal-dir",
        type=str,
        required=True,
        help="Path to saliency result directory.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to processed result directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(args)
    main(args)
