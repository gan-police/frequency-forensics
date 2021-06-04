"""
Code to load numpy files into memory for further processing
with PyTorch. Written with the numpy based data format
of https://github.com/RUB-SysSec/GANDCTAnalysis in mind.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "LoadNumpyDataset",
]


class LoadNumpyDataset(Dataset):
    """Create a data loader to load pre-processed numpy arrays
    into memory.
    """

    def __init__(self, data_dir: str, mean: float = None, std: float = None):
        """Create a Numpy-dataset object.

        Args:
            data_dir (str): A path to a pre-processed folder with numpy files.
            mean (float, optional): Pre-computed mean to normalize with.
                Defaults to None.
            std (float, optional): Pre-computed standard deviation to normalize
                with. Defaults to None.
        """
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob("./*.npy"))
        print("Loading ", data_dir)
        assert self.file_lst[-1].name == "labels.npy"
        self.labels = np.load(self.file_lst[-1])
        self.images = self.file_lst[:-1]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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
    """Compute dataset mean and standard deviation"""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate mean and std")
    parser.add_argument(
        "dir",
        type=str,
        help="path of training data for which mean and std are computed",
    )
    parser.add_argument(
        "-c",
        "--channelwise",
        action="store_true",
        help="calculate the mean and std for each channel of the data separately"
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
        if args.channelwise:
            axis = tuple(np.arange(len(img_data.shape[:-1])))

            # calculate mean and std in double to avoid precision problems
            mean = torch.mean(img_data.double(), axis).float()
            std = torch.std(img_data.double(), axis).float()
        else:
            # calculate mean and std in double to avoid precision problems
            mean = torch.mean(img_data.double()).float()
            std = torch.std(img_data.double()).float()

        return img_data, mean, std

    data, data_mean, data_std = compute_mean_std(data)

    print("mean", data_mean)
    print("std", data_std)

    norm = (data - data_mean) / data_std
    print("norm test", torch.mean(norm))
    print("std test", torch.std(norm))


if __name__ == "__main__":
    main()
