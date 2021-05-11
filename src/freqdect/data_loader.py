import torch
from pathlib import Path
import glob
import numpy as np
from torch.utils.data import Dataset


class LoadNumpyDataset(Dataset):
    def __init__(self, data_dir, mean=None, std=None):
        self.data_dir = data_dir
        self.file_lst = sorted(Path(data_dir).glob('./*.npy'))
        assert self.file_lst[-1].name == 'labels.npy'
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
        if self.mean:
            image = (image - self.mean) / self.std
        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {"image": image, "label": label}
        return sample


def main():
    import argparse
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Calculate mean and std')
    parser.add_argument('-r', '--raw', type=str, default="./data/source_data_raw_train",
                        help='path of raw training images (default: ./data/source_data_raw_train)')
    parser.add_argument('-p', '--packets', type=str, default="./data/source_data_packets_train",
                        help='path of wavelet packets of training data (default: ./data/source_data_packets_train)')

    args = parser.parse_args()

    print(args)

    # raw images - use only the training set.
    train_raw_set = LoadNumpyDataset(args.raw)
    # packets - use only the training set.
    train_packet_set = LoadNumpyDataset(args.packets)

    # train_dataloader = DataLoader(
    #     train_raw_set, batch_size=64, shuffle=True)
    # sample = next(iter(train_dataloader))
    # print(f"Feature batch shape: {sample['image'].size()}")
    # print(f"Labels batch shape: {sample['label'].size()}")
    # img = sample['image'][0].squeeze()
    # label = sample['label'][0]
    # plt.imshow(img.numpy().astype(np.uint8), cmap="gray")
    # plt.savefig('test_tmp.png')
    # print(f"Label: {label}")

    def compute_mean_std(data_set):
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

    # packets
    packet_loader = DataLoader(
        train_packet_set, batch_size=1, shuffle=True)
    packet_sample = next(iter(packet_loader))
    
    plt.plot(np.mean(np.reshape(packet_sample['image'][0].cpu().numpy(), [64, -1]), -1))
    plt.show()

    packet_data, packet_mean, packet_std = compute_mean_std(train_packet_set)
    print('packet mean', packet_mean)
    print('packet std', packet_std)

    # packet mean = 1.2623962
    # packet str = 3.023255
    norm = (packet_data - packet_mean) / packet_std
    print('packet norm test', np.mean(norm))
    print('packet std test', np.std(norm))
    del packet_data, norm

    # raw
    raw_data, raw_mean, raw_std = compute_mean_std(train_raw_set)
    print('raw mean', raw_mean)
    print('raw str', raw_std)

    # raw mean 112.52875
    # raw std 68.63312
    norm = (raw_data - raw_mean) / raw_std
    print('raw norm test', np.mean(norm))
    print('raw std test', np.std(norm))
    del raw_data, norm


if __name__ == '__main__':
    main()
