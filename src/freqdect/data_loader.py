import torch
import glob
import numpy as np
from torch.utils.data import Dataset


class LoadNumpyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_lst = glob.glob(data_dir + '/*.npy')
        self.file_lst.sort()
        assert self.file_lst[-1].split('/')[-1] == 'labels.npy'
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


def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    train_folder = './data_raw_train/'
    train_sample = np.load('./data_raw_train/000001.npy')
    labels = np.load('./data_raw_train/labels.npy')
    train_data_set = LoadNumpyDataset('./data_raw_train')
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
    print('str', np.std(img_data))
    
    # mean 112.52875
    # str 68.63312

    norm = (img_data - mean) / std
    print(np.mean(norm))
    print(np.std(norm))


if __name__ == '__main__':
    main()
