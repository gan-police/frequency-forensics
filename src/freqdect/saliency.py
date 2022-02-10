"""Sensitivity analysis/explainability module for trained models.
   Written by https://github.com/RaoulHeese ."""

import argparse
import os
import pickle


import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loader import LoadNumpyDataset
from .models import CNN, MLP, Regression


def save_to_disk(
    array_dict: dict,
    directory: str,
    sub_dir: str = "",
    counter: int = 0,
) -> None:
    """Save data batches (dicts of numpy arrays) to disk.

    Args:
        array_dict (dict): The data dict to store of the form {str: np.ndarray}.
        directory (str): The place to store the data.
        sub_dir (str): Subdirectory to store the images at.
        counter (int): Index of the file (sets filename).

    Returns:
        None
    """
    # loop over the batch dimension
    path = os.path.join(directory, sub_dir)
    if not os.path.exists(path):
        # print("creating", path)
        os.mkdir(path)
    with open(os.path.join(path, f"{counter:06}.npy"), "wb") as numpy_file:
        np.savez(numpy_file, **array_dict)


def saliency(
    model,
    data_loader: DataLoader,
    directory: str,
    sub_dir: str = "",
) -> int:
    """Calculate raw gradients (d out/d in) for each data point of a data set given one model.

    Args:
        model: PyTorch model to use for predictions.
        data_loader: DataLoader instance in which the data is stored.
        directory (str): The place to store the images at.
        sub_dir (str): Subdirectory to store the images at.

    Returns:
        int: The total number of processed gradients.
    """
    print(f"Evaluate gradients -> {sub_dir}...")

    if torch.cuda.is_available():
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    counter_total = 0
    counter_image = 0
    with tqdm(desc="process") as prog:
        for it, batch in enumerate(iter(data_loader)):

            model.eval()
            batch_images = batch["image"]
            # batch_labels = batch["label"]
            if torch.cuda.is_available():
                batch_images = batch_images.cuda(non_blocking=True)
                # batch_labels = batch_labels.cuda(non_blocking=True)
            batch_images.requires_grad = True
            out_batch = model(batch_images)
            n_inputs = out_batch.shape[0]
            n_classes = out_batch.shape[1]
            slc_batch = np.empty((n_inputs, n_classes) + batch_images[0, :].shape)

            image_index = []
            for i in range(n_inputs):
                for c in range(n_classes):
                    if batch_images.grad is not None:
                        batch_images.grad.zero_()
                    out_batch[i, c].backward(retain_graph=True)
                    slc = batch_images.grad[i].cpu().detach().numpy()
                    slc_batch[i, c, :] = slc
                    counter_total += 1
                image_index.append(counter_image)
                counter_image += 1

            slc_batch = np.asarray(slc_batch)  # [idx_in_batch, class_idx, data...]
            out_batch = out_batch.cpu().detach().numpy()  # [idx_in_batch, label]
            image_index = np.array(image_index)  # (index1, ..., indexN)
            save_to_disk(
                {"S": slc_batch, "O": out_batch, "I": image_index},
                directory,
                sub_dir,
                it,
            )

            prog.set_postfix(
                {"img": counter_image, "all": counter_total}, refresh=False
            )
            prog.update(1)

    return counter_image


def main(args):
    """Compute gradients (d out/d in) for trained models."""
    # torch seed for reproducible results
    torch.manual_seed(args.seed)

    # normalization
    if args.normalize:
        num_of_norm_vals = len(args.normalize)
        if (not num_of_norm_vals == 2) or (not num_of_norm_vals == 6):
            raise ValueError("Either two or six normalization values are required.")
        mean = torch.tensor(args.normalize[: (num_of_norm_vals // 2)])
        std = torch.tensor(args.normalize[(num_of_norm_vals // 2) :])
    elif args.calc_normalization:
        # load train data and compute mean and std
        try:
            with open(f"{args.data_prefix}_train/mean_std.pkl", "rb") as file:
                mean, std = pickle.load(file)
                mean = torch.from_numpy(mean.astype(np.float32))
                std = torch.from_numpy(std.astype(np.float32))
        except BaseException:
            print("loading mean and std from file failed. Re-computing.")
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

    # Load data
    train_data_set = LoadNumpyDataset(args.data_prefix + "_train", mean=mean, std=std)
    val_data_set = LoadNumpyDataset(args.data_prefix + "_val", mean=mean, std=std)
    test_data_set = LoadNumpyDataset(args.data_prefix + "_test", mean=mean, std=std)
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    test_data_loader = DataLoader(
        test_data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    # Build model
    if args.model == "mlp":
        model = MLP(args.nclasses).cuda()
    elif args.model == "cnn":
        model = CNN(args.nclasses, args.features == "packets").cuda()
    else:
        model = Regression(args.nclasses).cuda()

    # Load model parameters
    if torch.cuda.is_available():
        # saved on GPU, load on GPU
        model.load_state_dict(torch.load(args.model_pt_path))
    else:
        # saved on GPU, load on CPU
        map_location = torch.device("cpu")
        model.load_state_dict(torch.load(args.model_pt_path, map_location=map_location))
    print(f"Model loaded: {args.model_pt_path}")

    # Saliency
    directory = args.result_dir
    count = saliency(
        model, train_data_loader, directory, f"{args.model}_{args.features}_train"
    )
    print(f"Processed {count} train images.")
    count = saliency(
        model, test_data_loader, directory, f"{args.model}_{args.features}_test"
    )
    print(f"Processed {count} test images.")
    count = saliency(
        model, val_data_loader, directory, f"{args.model}_{args.features}_val"
    )
    print(f"Processed {count} val images.")

    print("Finished.")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-pt-path",
        type=str,
        required=True,
        help="Path to model pt file (required).",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Shared result dir (required).",
    )
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
        help="input batch size (default: 512)",
    )
    parser.add_argument(
        "--model",
        choices=["regression", "cnn", "mlp"],
        default="regression",
        help="The model type: regression, cnn, mlp. (default: regression).",
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
        help="calculates mean and standard deviation used in normalization"
        "from the training data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(args)
    main(args)
