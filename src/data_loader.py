import os
import torch
import glob
import numpy as np
from torch.utils.data import Dataset


class LoadNumpyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_lst = glob.glob(data_dir + '/*.npy')
        self.file_lst.sort()
        assert self.file_lst[-1].replace(os.sep, '/').split('/')[-1] == 'labels.npy'
        self.labels = np.load(self.file_lst[-1])
        self.images = self.file_lst[:-1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image.astype(np.float32))
        label = self.labels[idx]
        label = torch.tensor(int(label))
        sample = {"image": image, "label": label}
        return sample


if __name__ == '__main__':
    import argparse
    #import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Calculate mean and std')
    parser.add_argument('--train-data', type=str, default="./data/data_raw_train",
                        help='path of training data (default: ./data/data_raw_train)')
    args = parser.parse_args()

    print(args)

    train_sample = np.load(args.train_data + '/000001.npy')
    labels = np.load(args.train_data + '/labels.npy')
    train_data_set = LoadNumpyDataset(args.train_data)
    train_data_set.__getitem__(0)

    train_dataloader = DataLoader(
        train_data_set, batch_size=64, shuffle=True)
    sample = next(iter(train_dataloader))
    print(f"Feature batch shape: {sample['image'].size()}")
    print(f"Labels batch shape: {sample['label'].size()}")
    img = sample['image'][0].squeeze()
    label = sample['label'][0]
    # plt.imshow(img.numpy().astype(np.uint8), cmap="gray")
    # plt.savefig('test_tmp.png')
    print(f"Label: {label}")

    # compute mean and std
    img_lst = []
    for img_no in range(train_data_set.__len__()):
        img_lst.append(train_data_set.__getitem__(img_no)["image"].numpy())
    img_data = np.stack(img_lst, 0)

    mean = np.mean(img_data)
    std = np.std(img_data)
    print('mean', np.mean(img_data))
    print('std', np.std(img_data))
    
    # mean 112.52875
    # std 68.63312

    norm = (img_data - mean) / std
    print(np.mean(norm))
    print(np.std(norm))

