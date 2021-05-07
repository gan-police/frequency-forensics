import torch
import glob
import numpy as np
from torch.utils.data import Dataset


class LoadNumpyDataset(Dataset):
    def __init__(self, data_dir, mean=None, std=None):
        self.data_dir = data_dir
        self.file_lst = glob.glob(data_dir + '/*.npy')
        self.file_lst.sort()
        assert self.file_lst[-1].split('/')[-1] == 'labels.npy'
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
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # raw images.
    train_raw_set = LoadNumpyDataset('./data/source_data_raw_train')
    train_packet_set = LoadNumpyDataset('./data/source_data_packets_train')

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
            img_lst.append(data_set.__getitem__(img_no)["image"].numpy())
        img_data = np.stack(img_lst, 0)

        mean = np.mean(img_data)
        std = np.std(img_data)
        return img_data, mean, std

    # packets
    packet_loader = DataLoader(
        train_packet_set, batch_size=1, shuffle=True)
    packet_sample = next(iter(packet_loader))
    
    plt.plot(np.mean(np.reshape(packet_sample['image'][0].cpu().numpy(), [64, -1]), -1))
    plt.show()

    packet_data, packet_mean, packet_std = compute_mean_std(train_packet_set)
    print('packet mean', packet_mean)
    print('packet str', packet_std)

    # packet mean -
    # packet str -
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
