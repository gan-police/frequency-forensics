import argparse
import pickle
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from .data_loader import LoadNumpyDataset
from .models import CNN, Regression, MLP, compute_parameter_total
from torch.utils.tensorboard.writer import SummaryWriter


def val_test_loop(
    data_loader: DataLoader, model: torch.nn.Module, loss_fun
) -> Tuple[float, Any]:
    """Tests the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        loss_fun: The loss function, which is used to measure the loss of the model on the data set

    Returns:
        Tuple[float, Any]: The measured accuracy and loss of the model on the data set.
    """
    with torch.no_grad():
        model.eval()
        val_total = 0
        val_ok = 0
        for val_batch in iter(data_loader):
            batch_images = val_batch["image"].cuda(non_blocking=True)
            batch_labels = val_batch["label"].cuda(non_blocking=True)
            out = model(batch_images)
            val_loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += batch_labels.shape[0]
        val_acc = val_ok / val_total
        print("acc", val_acc, "ok", val_ok, "total", val_total)
    return val_acc, val_loss


def parse_args():
    """Parse cmd line args for training an image classifier"""

    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--features",
        choices=["raw", "packets"],
        default="packets",
        help="the representation type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="input batch size for testing (default: 512)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="weight decay for optimizer (default: 0)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs (default: 10)"
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=200,
        help="number of training steps after which the model is tested on the validation data set (default: 200)",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        default="./data/source_data_packets",
        help="shared prefix of the data paths (default: ./data/source_data_packets)",
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="the random seed pytorch works with."
    )

    parser.add_argument(
        "--model",
        choices=["regression", "cnn", "mlp"],
        default="regression",
        help="The model type chosse regression or CNN. Default: Regression.",
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )

    # one should not specify normalization parameters and request their calculation at the same time
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--normalize",
        nargs="+",
        type=float,
        metavar=("MEAN", "STD"),
        help="normalize with specified values for mean and standard deviation (either 2 or 6 values "
        "are accepted)",
    )
    group.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculates mean and standard deviation used in normalization from the training data",
    )
    return parser.parse_args()


def main():
    """Trains a model to classify images. All settings such as which model to use, parameters, normalization, data set path,
    seed etc. are specified via cmd line args.

    All training, validation and testing results are printed to stdout.
    After the training is done, the results are stored in a pickle dump in the 'log' folder.
    The state_dict of the trained model is stored there as well.
    """
    args = parse_args()
    print(args)

    # fix the seed in the interest of reproducible results.
    torch.manual_seed(args.seed)

    if args.normalize:
        num_of_norm_vals = len(args.normalize)
        assert num_of_norm_vals == 2 or num_of_norm_vals == 6
        mean = torch.tensor(args.normalize[: num_of_norm_vals // 2])
        std = torch.tensor(args.normalize[(num_of_norm_vals // 2) :])
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
        del img_data
    else:
        mean = None
        std = None

    print("mean", mean, "std", std)

    train_data_set = LoadNumpyDataset(args.data_prefix + "_train", mean=mean, std=std)
    val_data_set = LoadNumpyDataset(args.data_prefix + "_val", mean=mean, std=std)
    test_data_set = LoadNumpyDataset(args.data_prefix + "_test", mean=mean, std=std)

    train_data_loader = DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_data_loader = DataLoader(
        val_data_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

    if args.model == "mlp":
        model = MLP(args.nclasses).cuda()
    elif args.model == "cnn":
        model = CNN(args.nclasses, args.features == "packets").cuda()
    else:
        model = Regression(args.nclasses).cuda()

    print("model parameter count:", compute_parameter_total(model))

    if args.tensorboard:
        writer = SummaryWriter()

    loss_fun = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for e in range(args.epochs):
        # iterate over training data.
        for it, batch in enumerate(iter(train_data_loader)):
            model.train()
            optimizer.zero_grad()
            batch_images = batch["image"].cuda(non_blocking=True)
            batch_labels = batch["label"].cuda(non_blocking=True)

            out = model(batch_images)
            loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            acc = torch.sum(ok_mask.type(torch.float32)) / len(batch_labels)

            if it % 10 == 0:
                print(
                    "e",
                    e,
                    "it",
                    it,
                    "total",
                    step_total,
                    "loss",
                    loss.item(),
                    "acc",
                    acc.item(),
                )
            loss.backward()
            optimizer.step()
            step_total += 1
            loss_list.append([step_total, e, loss.item()])
            accuracy_list.append([step_total, e, acc.item()])

            if args.tensorboard:
                writer.add_scalar("train_loss", loss.item(), step_total)
                if it == 0:
                    writer.add_graph(model, batch_images)

            # iterate over val batches.
            if step_total % args.validation_interval == 0:
                print("validating....")
                val_acc, val_loss = val_test_loop(val_data_loader, model, loss_fun)
                validation_list.append([step_total, e, val_acc])
                if validation_list[-1] == 1.0:
                    print("val acc ideal stopping training.")
                    break

                if args.tensorboard:
                    writer.add_scalar("validation_loss", val_loss, step_total)
                    writer.add_scalar("validation_accuracy", val_acc, step_total)

        if args.tensorboard:
            writer.add_scalar("epochs", e, step_total)

    print(validation_list)

    # Run over the test set.
    print("Training done testing....")
    test_data_loader = DataLoader(
        test_data_set, args.batch_size, shuffle=False, num_workers=2
    )
    with torch.no_grad():
        test_acc, test_loss = val_test_loop(test_data_loader, model, loss_fun)
        print("test acc", test_acc)

    if args.tensorboard:
        writer.add_scalar("test_accuracy", test_acc, step_total)
        writer.add_scalar("test_loss", test_loss, step_total)

    log_name = "./log/" + args.data_prefix.split("/")[-1] + "_" + str(args.model)
    stats_file = log_name + ".pkl"
    try:
        res = pickle.load(open(stats_file, "rb"))
    except (OSError, IOError) as e:
        res = []
        print(
            e,
            "stats.pickle does not exist, \
              creating a new file.",
        )
    res.append(
        {
            "train_loss": loss_list,
            "train_acc": accuracy_list,
            "val_acc": validation_list,
            "test_acc": test_acc,
            "args": args,
            "iterations_per_epoch": len(iter(train_data_loader)),
        }
    )
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, " saved.")
    torch.save(model.state_dict(), log_name + ".pt")


if __name__ == "__main__":
    main()
