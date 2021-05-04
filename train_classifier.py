import torch
import pickle
import argparse
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
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('--features', choices=['raw', 'packets'],
                        default='packets',
                        help='the representation type')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='input batch size for testing (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='input batch size for testing (default: 0)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='input batch size for testing (default: 60)')
    args = parser.parse_args()
    print(args)

    train_data_set = LoadNumpyDataset('./data_raw_train')
    val_data_set = LoadNumpyDataset('./data_raw_val')
    test_data_set = LoadNumpyDataset('./data_raw_test')
    train_data_loader = DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True,
        num_workers=2)
    val_data_loader = DataLoader(
        val_data_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2)
    test_data_loader = DataLoader(
        test_data_set, args.batch_size, shuffle=False,
        num_workers=2)

    if args.features == 'packets':
        packets = True
    else:
        packets = False

    validation_list = []
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
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    for e in range(args.epochs):
        # iterate over training data.
        for it, batch in enumerate(iter(train_data_loader)):
            optimizer.zero_grad()
            batch_images = batch['image'].cuda(non_blocking=True)
            batch_labels = batch['label'].cuda(non_blocking=True)
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

            if it % 10 == 0:
                print('e', e, 'it', it, 'loss', loss.item(), 'acc', acc.item())
            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append((step_total, loss.item()))
            accuracy_list.append([step_total, acc.item()])

            # iterate over val batches.
            if step_total % 100 == 0:
                print('validating....')
                with torch.no_grad():
                    val_total = 0
                    val_ok = 0
                    for val_batch in iter(val_data_loader):
                        batch_images = val_batch['image'].cuda(non_blocking=True)
                        batch_labels = val_batch['label'].cuda(non_blocking=True)
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
                        val_ok += torch.sum(ok_mask).item()
                        val_total += batch_labels.shape[0]
                    validation_list.append((step_total, val_ok / val_total))
                    print('val acc', validation_list[-1],
                          'val_ok', val_ok,
                          'total', val_total)
                    if validation_list[-1] == 1.:
                        print('val acc ideal stopping training.')
                        break
    print(validation_list)

    # Run over the test set.
    print('Training done testing....')
    with torch.no_grad():
        test_total = 0
        test_ok = 0
        for test_batch in iter(test_data_loader):
            batch_images = test_batch['image'].cuda(non_blocking=True)
            batch_labels = test_batch['label'].cuda(non_blocking=True)
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
        test_acc = test_ok / test_total
        print('test acc', test_acc,
              'test_ok', test_ok,
              'total', test_total)

    stats_file = './log/v2' + 'packets' + str(packets) + '.pkl'
    try:
        res = pickle.load(open(stats_file, "rb"))
    except (OSError, IOError) as e:
        res = []
        print(e, 'stats.pickle does not exist, \
              creating a new file.')

    res.append({'train loss': loss_list,
                'train acc': accuracy_list,
                'val acc': validation_list,
                'test acc': test_acc,
                'args': args,
                'model': model})
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, ' saved.')
