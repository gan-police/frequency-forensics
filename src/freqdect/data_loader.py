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
    "NumpyDataset",
    "CombinedDataset"
]


class NumpyDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays into memory."""

    def __init__(
        self, data_dir: str, mean: Optional[float] = None, std: Optional[float] = None,
        key: Optional[str] = 'image'
    ):
        """Create a Numpy-dataset object.

        Args:
            data_dir: A path to a pre-processed folder with numpy files.
            mean: Pre-computed mean to normalize with. Defaults to None.
            std: Pre-computed standard deviation to normalize with. Defaults to None.
            key: The key for the input or 'x' component of the dataset.
                Defaults to "image".

        Raises:
            ValueError: If an unexpected file name is given
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
        self.key = key

    def __len__(self) -> int:
        """Return the data set length."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Get a dataset element.

        Args:
            idx (int): The element index of the data pair to return.

        Returns:
            [dict]: Returns a dictionary with the self.key
                    default ("image") and "label" keys.
        """
        img_path = self.images[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image.astype(np.float32))
        # normalize the data.
        if self.mean is not None:
            image = (image - self.mean) / self.std
        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {self.key: image, "label": label}
        return sample


class CombinedDataset(Dataset):
    def __init__(self, sets: list):
        """Create an merged dataset, combining many numpy datasets.

        Args:
            sets (list): A list of NumpyDataset objects.
        """        
        self.sets = sets
        self.len = len(sets[0])
        assert not any(self.len != len(s) for s in sets)

    @property
    def key(self) -> list:
        return [d.key for d in self.sets]   

    def __len__(self) -> int:
        """Return the data set length."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        label_list = [s.__getitem__(idx)["label"] for s in self.sets]
        # the labels should all be the same
        assert not any([label_list[0] != l for l in label_list])
        label = label_list[0]
        dict = {set.key: set.__getitem__(idx)[set.key] for set in self.sets}
        dict["label"] = label
        return dict


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

    data = NumpyDataset(args.dir)

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
    with open(file_name, "wb") as f:
        pickle.dump([mean.numpy(), std.numpy()], f)
    print("stored in", file_name)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path1 = "/nvme/mwolter/celeba/celeba_align_png_cropped_raw_train"
    path2 = "/nvme/mwolter/celeba/celeba_align_png_cropped_log_fourier_haar_reflect_3_train"

    data1 = NumpyDataset(path1, key='raw')
    data2 = NumpyDataset(path2, key='fft')


    data = CombinedDataset([data1, data2])
    item = data.__getitem__(0)

    for no in range(len(data)):
        item = data.__getitem__(no)
        fft = torch.log(torch.abs(
            torch.fft.fft(item['raw'][..., 0])) + 1e-12)
        fft2 = item['fft'][..., 0]
        print("{:2.2f}".format(torch.max(torch.abs(fft - fft2)).item()))

    print('stop')

    # main()
