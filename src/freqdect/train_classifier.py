import torch
import pickle
import argparse
import numpy as np
from .data_loader import LoadNumpyDataset
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
            # batch_images = (batch_images - 112.52875) / 68.63312
            out = model(batch_images)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += batch_labels.shape[0]
        val_acc = val_ok / val_total
        print('acc', val_acc,
              'ok', val_ok,
              'total', val_total)
    return val_acc


def main():
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('--features', choices=['raw', 'packets'],
                        default='packets',
                        help='the representation type')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate for optimizer (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay for optimizer (default: 0)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs (default: 10)')
    parser.add_argument('--data-prefix', type=str, default="./data/source_data",
                        help='shared prefix of the data paths (default: ./data/source_data)')
    parser.add_argument('--nclasses', type=int, default=2,
                        help='number of classes (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='the random seed pytorch works with.')

    # one should not specify normalization parameters and request their calculation at the same time
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--normalize', nargs='+', type=float, metavar=('MEAN', 'STD'),
                       help='normalize with specified values for mean and standard deviation (either 2 or 6 values '
                            'are accepted)')
    group.add_argument('--calc-normalization', action='store_true',
                       help='calculates mean and standard deviation used in normalization from the training data')
    args = parser.parse_args()
    print(args)

    # fix the seed in the interest of reproducible results.
    torch.manual_seed(args.seed)

    if args.features == 'packets':
        packets = True
        default_mean = torch.cuda.FloatTensor([1.2739, 1.2591, 1.2542])
        default_std = torch.cuda.FloatTensor([3.0472, 2.9926, 3.0297])
    elif args.features == 'raw':
        packets = False
        default_mean = torch.cuda.FloatTensor([132.6314, 108.3550, 96.8289])
        default_std = torch.cuda.FloatTensor([71.1634, 64.5999, 64.9532])
    else:
        raise NotImplementedError

    if args.normalize:
        num_of_norm_vals = len(args.normalize)
        assert num_of_norm_vals == 2 or num_of_norm_vals == 6
        mean = torch.cuda.FloatTensor(args.normalize[:num_of_norm_vals // 2])
        std = torch.cuda.FloatTensor(args.normalize[(num_of_norm_vals // 2):])
    elif args.calc_normalization:
        # load train data and compute mean and std
        train_data_set = LoadNumpyDataset(args.data_prefix + "_train")

        img_lst = []
        for img_no in range(train_data_set.__len__()):
            img_lst.append(train_data_set.__getitem__(img_no)["image"])
        img_data = torch.stack(img_lst, 0)

        # average all axis except the color channel
        axis = tuple(np.arange(len(img_data.shape[:-1])))

        # calculate mean and std in double to avoid precision problems
        mean = torch.mean(img_data.double(), axis).float()
        std = torch.std(img_data.double(), axis).float()
    else:
        mean = default_mean
        std = default_std

    print("mean", mean, "std", std)

    train_data_set = LoadNumpyDataset(
        args.data_prefix + '_train', mean=mean, std=std)
    val_data_set = LoadNumpyDataset(
        args.data_prefix + '_val', mean=mean, std=std)
    test_data_set = LoadNumpyDataset(
        args.data_prefix + '_test', mean=mean, std=std)

    train_data_loader = DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True,
        num_workers=15)
    val_data_loader = DataLoader(
        val_data_set, batch_size=args.batch_size, shuffle=False,
        num_workers=15)

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

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

            out = model(batch_images)
            loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            acc = torch.sum(ok_mask.type(torch.float32)) / len(batch_labels)

            if it % 10 == 0:
                print('e', e, 'it', it, 'loss', loss.item(), 'acc', acc.item())
            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append([step_total, loss.item()])
            accuracy_list.append([step_total, acc.item()])

            # iterate over val batches.
            if step_total % 100 == 0:
                print('validating....')
                validation_list.append(
                    [step_total, val_test_loop(val_data_loader, model, loss_fun)])
                if validation_list[-1] == 1.:
                    print('val acc ideal stopping training.')
                    break
    print(validation_list)

    # Run over the test set.
    print('Training done testing....')
    test_data_loader = DataLoader(
        test_data_set, args.batch_size, shuffle=False,
        num_workers=15)
    with torch.no_grad():
        test_acc = val_test_loop(test_data_loader, model, loss_fun)
        print('test acc', test_acc)

    stats_file = './log/' + 'packets' + str(packets) + '.pkl'
    try:
        res = pickle.load(open(stats_file, "rb"))
    except (OSError, IOError) as e:
        res = []
        print(e, 'stats.pickle does not exist, \
              creating a new file.')

    res.append({'train_loss': loss_list,
                'train_acc': accuracy_list,
                'val_acc': validation_list,
                'test_acc': test_acc,
                'args': args})
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, ' saved.')


if __name__ == '__main__':
    main()
