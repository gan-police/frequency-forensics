"""Source code to train deepfake detectors in wavelet and pixel space."""

import argparse
import os
import pickle
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .data_loader import CombinedDataset, NumpyDataset
from .models import CNN, Regression, compute_parameter_total, save_model


def val_test_loop(
    data_loader,
    model: torch.nn.Module,
    loss_fun,
    make_binary_labels: bool = False,
    _description: str = "Validation",
    pbar: bool = False,
) -> Tuple[float, Any]:
    """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

    Args:
        data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
            e.g. a test or validation set in a data split.
        model (torch.nn.Module): The model to evaluate.
        loss_fun: The loss function, which is used to measure the loss of the model on the data set
        make_binary_labels (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the label 0 encodes 'real'. All other labels are cosidered fake data, and are set to 1.

    Returns:
        Tuple[float, Any]: The measured accuracy and loss of the model on the data set.
    """
    with torch.no_grad():
        model.eval()
        val_total = 0
        val_ok = 0
        for val_batch in iter(data_loader):
            if type(data_loader.dataset) is CombinedDataset:
                batch_images = {
                    key: val_batch[key].cuda(non_blocking=True)
                    for key in data_loader.dataset.key
                }
            else:
                batch_images = val_batch[data_loader.dataset.key].cuda(
                    non_blocking=True
                )
            batch_labels = val_batch["label"].cuda(non_blocking=True)
            out = model(batch_images)
            if make_binary_labels:
                batch_labels[batch_labels > 0] = 1
            val_loss = loss_fun(torch.squeeze(out), batch_labels)
            ok_mask = torch.eq(torch.max(out, dim=-1)[1], batch_labels)
            val_ok += torch.sum(ok_mask).item()
            val_total += batch_labels.shape[0]
        val_acc = val_ok / val_total
        print("acc", val_acc, "ok", val_ok, "total", val_total)
    return val_acc, val_loss


def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--features",
        choices=["raw", "packets", "all-packets", "fourier", "all-packets-fourier"],
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
        "--epochs", type=int, default=20, help="number of epochs (default: 10)"
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
        nargs="+",
        default=["./data/source_data_packets"],
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
        choices=["regression", "cnn"],
        default="regression",
        help="The model type chosse regression or CNN. Default: Regression.",
    )

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="enables a tensorboard visualization.",
    )

    parser.add_argument(
        "--pbar",
        action="store_true",
        help="enables progress bars",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes started by the test and validation data loaders. The training data_loader "
        "uses three times this argument many workers. Hence, this argument should probably be chosen below 10. "
        "Defaults to 2.",
    )

    parser.add_argument(
        "--class-weights",
        type=float,
        metavar="CLASS_WEIGHT",
        nargs="+",
        default=None,
        help="If specified, training samples are weighted based on their class "
        "in the loss calculation. Expects one weight per class.",
    )

    # one should not specify normalization parameters and request their calculation at the same time
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculates mean and standard deviation used in normalization"
        "from the training data",
    )
    return parser.parse_args()


def create_data_loaders(data_prefix: str, batch_size: int) -> tuple:
    """Create the data loaders needed for training.

    The test set is created outside a loader.

    Args:
        data_prefix (str): Where to look for the data.

    Raises:
        RuntimeError: Raised if the prefix is incorrect.

    Returns:
        list: train_data_loader, val_data_loader, test_data_set

    # noqa: DAR401
    """
    data_set_list = []
    for data_prefix_el in data_prefix:
        with open(f"{data_prefix_el}_train/mean_std.pkl", "rb") as file:
            mean, std = pickle.load(file)
            mean = torch.from_numpy(mean.astype(np.float32))
            std = torch.from_numpy(std.astype(np.float32))

        print("mean", mean, "std", std)
        key = "image"
        if "raw" in data_prefix_el.split("_"):
            key = "raw"
        elif "packets" in data_prefix_el.split("_"):
            key = "packets" + data_prefix_el.split("_")[-1]
        elif "fourier" in data_prefix_el.split("_"):
            key = "fourier"

        train_data_set = NumpyDataset(
            data_prefix_el + "_train", mean=mean, std=std, key=key
        )
        val_data_set = NumpyDataset(
            data_prefix_el + "_val", mean=mean, std=std, key=key
        )
        test_data_set = NumpyDataset(
            data_prefix_el + "_test", mean=mean, std=std, key=key
        )
        data_set_list.append((train_data_set, val_data_set, test_data_set))

    if len(data_set_list) == 1:
        train_data_loader = DataLoader(
            train_data_set, batch_size=batch_size, shuffle=True, num_workers=3
        )
        val_data_loader = DataLoader(
            val_data_set, batch_size=batch_size, shuffle=False, num_workers=3
        )
        test_data_sets: Any = test_data_set
    elif len(data_set_list) > 1:
        train_data_sets = [el[0] for el in data_set_list]
        val_data_sets = [el[1] for el in data_set_list]
        test_data_sets = [el[2] for el in data_set_list]
        train_data_loader = DataLoader(
            CombinedDataset(train_data_sets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
        )
        val_data_loader = DataLoader(
            CombinedDataset(val_data_sets),
            batch_size=batch_size,
            shuffle=False,
            num_workers=3,
        )
    else:
        raise RuntimeError("Failed to load data from the specified prefixes.")

    return train_data_loader, val_data_loader, test_data_sets


def main():
    """Trains a model to classify images.

    All settings such as which model to use, parameters, normalization, data set path,
    seed etc. are specified via cmd line args.
    All training, validation and testing results are printed to stdout.
    After the training is done, the results are stored in a pickle dump in the 'log' folder.
    The state_dict of the trained model is stored there as well.

    Raises:
        ValueError: Raised if mean and std values are incomplete or if the number of
            specified class weights does not match the number of classes.

    # noqa: DAR401
    """
    args = _parse_args()
    print(args)

    if args.class_weights and len(args.class_weights) != args.nclasses:
        raise ValueError(
            f"The number of class_weights ({len(args.class_weights)}) must equal "
            f"the number of classes ({args.nclasses})"
        )

    # fix the seed in the interest of reproducible results.
    torch.manual_seed(args.seed)

    make_binary_labels = args.nclasses == 2
    train_data_loader, val_data_loader, test_data_set = create_data_loaders(
        args.data_prefix, args.batch_size
    )

    validation_list = []
    loss_list = []
    accuracy_list = []
    step_total = 0

    if args.model == "cnn":
        model = CNN(args.nclasses, args.features).cuda()
    else:
        model = Regression(args.nclasses).cuda()

    print("model parameter count:", compute_parameter_total(model))

    if args.tensorboard:
        writer_str = "runs/"
        writer_str += "params_test2/"
        writer_str += f"{args.model}/"
        writer_str += f"{args.batch_size}/"
        writer_str += str(args.data_prefix.split("/")[-1]) + "/"
        writer_str += f"{args.learning_rate}_"
        writer_str += f"{args.seed}"
        writer = SummaryWriter(writer_str, max_queue=100)

    if args.class_weights:
        loss_fun = torch.nn.NLLLoss(weight=torch.tensor(args.class_weights).cuda())
    else:
        loss_fun = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for e in tqdm(
        range(args.epochs), desc="Epochs", unit="epochs", disable=not args.pbar
    ):
        # iterate over training data.
        for it, batch in enumerate(
            tqdm(
                iter(train_data_loader),
                desc="Training",
                unit="batches",
                disable=not args.pbar,
            )
        ):
            model.train()
            optimizer.zero_grad()
            # find the bug.
            if type(train_data_loader.dataset) is CombinedDataset:
                batch_images = {
                    key: batch[key].cuda(non_blocking=True)
                    for key in train_data_loader.dataset.key
                }
            else:
                batch_images = batch[train_data_loader.dataset.key].cuda(
                    non_blocking=True
                )

            batch_labels = batch["label"].cuda(non_blocking=True)
            if make_binary_labels:
                batch_labels[batch_labels > 0] = 1

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
                writer.add_scalar("loss/train", loss.item(), step_total)
                writer.add_scalar("accuracy/train", acc.item(), step_total)
                if step_total == 0:
                    writer.add_graph(model, batch_images)

            # iterate over val batches.
            if step_total % args.validation_interval == 0:
                val_acc, val_loss = val_test_loop(
                    val_data_loader,
                    model,
                    loss_fun,
                    make_binary_labels=make_binary_labels,
                    pbar=args.pbar,
                )
                validation_list.append([step_total, e, val_acc])
                if validation_list[-1] == 1.0:
                    print("val acc ideal stopping training.")
                    break

                if args.tensorboard:
                    writer.add_scalar("loss/validation", val_loss, step_total)
                    writer.add_scalar("accuracy/validation", val_acc, step_total)

        if args.tensorboard:
            writer.add_scalar("epochs", e, step_total)

    print(validation_list)

    if not os.path.exists("./log/"):
        os.makedirs("./log/")
    model_file = (
        "./log/"
        + args.data_prefix.split("/")[-1]
        + "_"
        + str(args.learning_rate)
        + "_"
        + f"{args.epochs}e"
        + "_"
        + str(args.model)
    )
    save_model(model, model_file + "_" + str(args.seed) + ".pt")
    print(model_file, " saved.")

    # Run over the test set.
    print("Training done testing....")
    if type(test_data_set) is list:
        test_data_set = CombinedDataset(test_data_set)

    test_data_loader = DataLoader(
        test_data_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    with torch.no_grad():
        test_acc, test_loss = val_test_loop(
            test_data_loader,
            model,
            loss_fun,
            make_binary_labels=make_binary_labels,
            pbar=not args.pbar,
            _description="Testing",
        )
        print("test acc", test_acc)

    if args.tensorboard:
        writer.add_scalar("accuracy/test", test_acc, step_total)
        writer.add_scalar("loss/test", test_loss, step_total)

    _save_stats(
        model_file,
        loss_list,
        accuracy_list,
        validation_list,
        test_acc,
        args,
        len(iter(train_data_loader)),
    )

    if args.tensorboard:
        writer.close()


def _save_stats(
    model_file: str,
    loss_list: list,
    accuracy_list: list,
    validation_list: list,
    test_acc: float,
    args,
    iterations_per_epoch: int,
):
    stats_file = model_file + ".pkl"
    try:
        res = pickle.load(open(stats_file, "rb"))
    except OSError as e:
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
            "iterations_per_epoch": iterations_per_epoch,
        }
    )
    pickle.dump(res, open(stats_file, "wb"))
    print(stats_file, " saved.")


if __name__ == "__main__":
    main()
