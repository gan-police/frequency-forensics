"""Sensitivity analysis/explainability module for trained models."""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle

from .data_loader import LoadNumpyDataset
from .models import CNN, MLP, Regression


def save_to_disk(
    data_batch: np.ndarray,
    directory: str,
    dir_suffix: str = "",
    counter: int = 0,
    ):
    """Save data batches (numpy arrays) to disk.

    Args:
        data_batch (np.ndarray): The data batch to store.
        directory (str): The place to store the data.
        dir_suffix (str): A comment which is attatched to the output directory.
        counter (int): Index of the file (sets filename).

    Returns:
        None
    """
    # loop over the batch dimension
    path = f"{directory}_{dir_suffix}"
    if not os.path.exists(path):
        print("creating", path)
        os.mkdir(path)
    with open(os.path.join(path, f"{counter:06}.npy"), "wb") as numpy_file:
        np.save(numpy_file, data_batch)


def saliency(model,
             data_loader,
             directory: str,
             dir_suffix: str = "",
             ) -> int:
    """Save images to disk using their position on the dataset as filename.

    Args:
        model: PyTorch model to use for predictions.
        data_loader: 
        directory (str): The place to store the images at.
        dir_suffix (str): A comment which is attatched to the output directory.

    Returns:
        int: The total number of processed gradients.
    """
    if torch.cuda.is_available():
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    counter = 0
    for it, batch in enumerate(iter(data_loader)):
        model.eval()
        batch_images = batch["image"]
        batch_labels = batch["label"]
        if torch.cuda.is_available():
            batch_images = batch_images.cuda(non_blocking=True)
            #batch_labels = batch_labels.cuda(non_blocking=True)
        batch_images.requires_grad = True
        out = model(batch_images)
        n_inputs = out.shape[0]
        n_classes = out.shape[1]
        slc_batch = np.empty((n_inputs, n_classes) + batch_images[0,:].shape)
        print(slc_batch.shape)
        for i in range(n_inputs):
            for c in range(n_classes):
                if batch_images.grad is not None:
                    batch_images.grad.zero_()
                out[i, c].backward(retain_graph=True)
                slc = batch_images.grad[i].cpu().detach().numpy()
                slc_batch[i,c,:] = slc
                counter += 1
        slc_batch = np.asarray(slc_batch)  # [idx_in_batch, class_idx, data...]
        save_to_disk(slc_batch, directory, dir_suffix, it)

    return counter


def main(args):
    # torch seed for reproducible results
    torch.manual_seed(args.seed)

    # normalization
    if args.calc_normalization:
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
    train_data_set = LoadNumpyDataset(args.data_prefix + "_train", mean=mean, std=std, drop_last=False)
    val_data_set = LoadNumpyDataset(args.data_prefix + "_val", mean=mean, std=std, drop_last=False)
    test_data_set = LoadNumpyDataset(args.data_prefix + "_test", mean=mean, std=std, drop_last=False)
    train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_data_loader = DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_data_loader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build model
    if args.model == "mlp":
        model = MLP(args.nclasses).cuda()
    elif args.model == "cnn":
        model = CNN(args.nclasses, args.features == "packets").cuda()
    else:
        model = Regression(args.nclasses).cuda()

    # Load model parameters
    if torch.cuda.is_available():
        map_location = torch.cuda.current_device()
    else:
        map_location = torch.device('cpu')
    model.load_state_dict(torch.load(args.model_pt_path, map_location=map_location))

    # Saliency
    directory = args.result_dir
    count = saliency(model, train_data_loader, directory, "train")
    print(f"Processed {count} train images.")
    count = saliency(model, test_data_loader, directory, "test")
    print(f"Processed {count} test images.")
    count = saliency(model, val_data_loader, directory, "val")
    print(f"Processed {count} val images.")


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
    parser.add_argument(
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
