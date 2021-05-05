import torch
import pickle
import argparse
import numpy as np
from src.wavelet_math import compute_pytorch_packet_representation_2d_tensor
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


def val_test_loop(data_loader, model, loss_fun):
    with torch.no_grad():
        val_total = 0
        val_ok = 0
        for val_batch in iter(data_loader):
            batch_images = val_batch['image'].cuda(non_blocking=True)
            batch_labels = val_batch['label'].cuda(non_blocking=True)
            # batch_labels = torch.nn.functional.one_hot(batch_labels)
            batch_images = (batch_images - 112.52875) / 68.63312
            if packets:
                channel_list = []
                for channel in range(3):
                    channel_list.append(
                        compute_pytorch_packet_representation_2d_tensor(
                            batch_images[:, :, :, channel],
                            wavelet_str=wavelet, max_lev=max_lev))
                batch_images = torch.stack(channel_list, -1)
            out = model(batch_images)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += batch_labels.shape[0]
        val_acc = val_ok / val_total
        print('acc', val_acc,
              'ok', val_ok,
              'total', val_total)
    return val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('--features', choices=['raw', 'packets'],
                        default='packets',
                        help='the representation type')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate for optimizer (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='learning rate for optimizer (default: 0)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs (default: 60)')
    parser.add_argument('--data-prefix', type=str, default="./data/data_raw",
                        help='shared prefix of the data paths (default: ./data/data_raw)')
    parser.add_argument('--nclasses', type=int, default=2,
                        help='number of classes (default: 2)')

    # one should not specify normalization parameters and request their calculation at the same time
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--normalize', nargs='+', type=float, metavar=('MEAN', 'STD'),
                       help='normalize with specified values for mean and standard deviation (either 2 or 6 values are accepted)')
    group.add_argument('--calc-normalization', action='store_true',
                       help='calculates mean and standard deviation used in normalization from the training data')
    args = parser.parse_args()
    print(args)

    train_data_set = LoadNumpyDataset(args.data_prefix + "_train")
    val_data_set = LoadNumpyDataset(args.data_prefix + "_val")
    test_data_set = LoadNumpyDataset(args.data_prefix + "_test")

    train_data_loader = DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True,
        num_workers=1)
    val_data_loader = DataLoader(
        val_data_set, batch_size=args.batch_size, shuffle=False,
        num_workers=1)

    if args.features == 'packets':
        packets = True
    else:
        packets = False

    if args.normalize:
        num_of_norm_vals = len(args.normalize)
        assert num_of_norm_vals == 2 or num_of_norm_vals == 6
        mean = torch.cuda.FloatTensor(args.normalize[:num_of_norm_vals//2]).cuda()
        std = torch.cuda.FloatTensor(args.normalize[num_of_norm_vals//2:]).cuda()
    elif args.calc_normalization:
        # compute mean and std
        img_lst = []
        for img_no in range(train_data_set.__len__()):
            img_lst.append(train_data_set.__getitem__(img_no)["image"])
        img_data = torch.stack(img_lst, 0)

        # average all axis except the color channel
        axis = tuple(np.arange(len(img_data.shape[:-1])))

        # calculate mean and std in double to avoid precision problems
        mean = torch.mean(img_data.double(), axis).float().cuda()
        std = torch.std(img_data.double(), axis).float().cuda()
    else:
        mean, std = (112.52875, 68.63312)

    print("mean", mean, "std", std)

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

    if packets:
        wavelet = 'db1'
        max_lev = 3
    model = Regression(49152, args.nclasses).cuda()

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

            # normalize image data
            batch_images = (batch_images - mean) / std
            if packets:
                channel_list = []
                for channel in range(3):
                    channel_list.append(
                        compute_pytorch_packet_representation_2d_tensor(
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
                validation_list.append(
                    val_test_loop(val_data_loader, model, loss_fun))
                if validation_list[-1] == 1.:
                    print('val acc ideal stopping training.')
                    break
    print(validation_list)

    # Run over the test set.
    print('Training done testing....')
    test_data_loader = DataLoader(
        test_data_set, args.batch_size, shuffle=False,
        num_workers=2)
    with torch.no_grad():
        test_acc = val_test_loop(test_data_loader, model, loss_fun)
        print('test acc', test_acc)

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
