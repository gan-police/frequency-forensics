"""
Code to load numpy files into memory for further processing with PyTorch.

Written with the numpy based data format
of https://github.com/RUB-SysSec/GANDCTAnalysis in mind.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "LoadNumpyDataset",
]


class LoadNumpyDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self, data_dir: str, mean: Optional[float] = None, std: Optional[float] = None
    ):
        """Create a Numpy-dataset object.

        :param data_dir: A path to a pre-processed folder with numpy files.
        :param mean: Pre-computed mean to normalize with. Defaults to None.
        :param std: Pre-computed standard deviation to normalize with. Defaults to None.
        :raises ValueError: If an unexpected file name is given
        """
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob("./*.npy"))
        print("Loading ", data_dir)
        if self.file_lst[-1].name != "labels.npy":
            raise ValueError("unexpected file name")
        self.labels = np.load(self.file_lst[-1])
        self.images = self.file_lst[:-1]
        self.mean = mean
        self.std = std

    def __len__(self):
        """Return the data set length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            [dict]: Returns a dictionary with the "image" and label "keys".
        """
        img_path = self.images[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image.astype(np.float32))
        # normalize the data.
        if self.mean is not None:
            image = (image - self.mean) / self.std
        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {"image": image, "label": label}
        return sample


def main():
    """Compute dataset mean and standard deviation and store it."""
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Calculate mean and std")
    parser.add_argument(
        "dir",
        type=str,
        help="path of training data for which mean and std are computed",
    )
    args = parser.parse_args()

    print(args)

    data = LoadNumpyDataset(args.dir)

    def compute_mean_std(data_set: Dataset) -> tuple:
        """Compute mean and stad values by looping over a dataset.

        Args:
            data_set (Dataset): A torch style dataset.

        Returns:
            tuple: the raw_data, as well as mean and std values.
        """
        # compute mean and std
        img_lst = []
        for img_no in range(data_set.__len__()):
            img_lst.append(data_set.__getitem__(img_no)["image"])
        img_data = torch.stack(img_lst, 0)

        # average all axis except the color channel
        axis = tuple(np.arange(len(img_data.shape[:-1])))
        # calculate mean and std in double to avoid precision problems
        mean = torch.mean(img_data.double(), axis).float()
        std = torch.std(img_data.double(), axis).float()
        return img_data, mean, std

    data, mean, std = compute_mean_std(data)

    print("mean", mean)
    print("std", std)
    file_name = f"{args.dir}/mean_std.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump([mean.numpy(), std.numpy()], f)
    print('stored in {file_name}')

if __name__ == "__main__":
    main()
