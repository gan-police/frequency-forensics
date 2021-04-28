import torch
import pickle
import numpy as np
from src.wavelet_math import compute_pytorch_packet_representation_2d
from src.data_loader import LoadNumpyDataset
from torch.utils.data import DataLoader


class Regression(torch.nn.Module):

    def __init__(self, input_size, classes):
        super().__init__()
        self.linear = torch.nn.Linear(
            input_size, classes)

        # self.activation = torch.nn.Sigmoid()
        self.activation = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.activation(self.linear(x_flat))


if __name__ == '__main__':
    train_data_set = LoadNumpyDataset('./data_raw_train')
    val_data_set = LoadNumpyDataset('./data_raw_val')
    train_data_loader = DataLoader(
        train_data_set, batch_size=1024, shuffle=True,
        num_workers=1, pin_memory=True)
    val_data_loader = DataLoader(
        val_data_set, batch_size=1024, shuffle=False,
        num_workers=1, pin_memory=True)

    packets = False
    epochs = 4
    test_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

    if packets:
        wavelet = 'db2'
        max_lev = 3
        model = Regression(62208, 2).cuda()
    else:
        model = Regression(49152, 2).cuda()

    loss_fun = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for e in range(epochs):
        # iterate over training data.
        for it, batch in enumerate(iter(train_data_loader)):
            optimizer.zero_grad()
            batch_images = batch['image'].cuda(non_blocking=True)
            batch_labels = batch['label'].cuda(non_blocking=True)
            # batch_labels = torch.nn.functional.one_hot(batch_labels)
            batch_images = (batch_images - 112.52875) / 68.63312
            if packets:
                channel_list = []
                for channel in range(3):
                    channel_list.append(
                        compute_pytorch_packet_representation_2d(
                            batch_images[:, :, :, channel],
                            wavelet_str=wavelet, max_lev=max_lev))
                batch_images = torch.stack(channel_list, -1)

            out = model(batch_images)
            loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            acc = torch.sum(ok_mask.type(torch.float32)) / len(batch_labels)
            print('e', e, 'it', it, 'loss', loss.item(), 'acc', acc.item())
            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append((step_total, loss.item()))
            accuracy_list.append([step_total, acc.item()])

        # iterate over val batches.
        print('testing...')
        with torch.no_grad():
            test_total = 0
            test_ok = 0
            for test_batch in enumerate(iter(val_data_loader)):
                batch_images = batch['image'].cuda(non_blocking=True)
                batch_labels = batch['label'].cuda(non_blocking=True)
                # batch_labels = torch.nn.functional.one_hot(batch_labels)
                batch_images = (batch_images - 112.52875) / 68.63312
                if packets:
                    channel_list = []
                    for channel in range(3):
                        channel_list.append(
                            compute_pytorch_packet_representation_2d(
                                batch_images[:, :, :, channel],
                                wavelet_str=wavelet, max_lev=max_lev))
                    batch_images = torch.stack(channel_list, -1)

                out = model(batch_images)
                loss = loss_fun(torch.squeeze(out), batch_labels)
                ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
                test_ok += torch.sum(ok_mask).item()
                test_total += batch_labels.shape[0]
            test_list.append((step_total, test_ok / test_total))
            print('test acc', test_list[-1],
                  'test_ok', test_ok,
                  'total', test_total)
    print(test_list)
    with open('./log/' + 'packets' + str(packets) + '.pkl', 'wb') as f:
        pickle.dump([loss_list, accuracy_list, test_list], f)
