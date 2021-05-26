import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from .models import CNN, Regression, initialize_model
from .data_loader import LoadNumpyDataset


def calculate_confusion_matrix():
    parser = argparse.ArgumentParser(description="Calculate the confusion matrix")
    parser.add_argument("classifier_path", type=str, help="path to classifier model file")
    parser.add_argument("data", type=str, help="path of folder containing the test data")
    parser.add_argument(
        "model",
        choices=["regression", "CNN"],
        help="The model type. Choose regression or CNN."
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
        help="input batch size for testing (default: 512)",
    )
    parser.add_argument(
        "--normalize",
        nargs="+",
        type=float,
        metavar=("MEAN", "STD"),
        help="normalize with specified values for mean and standard deviation (either 2 or 6 values "
             "are accepted)",
    )
    args = parser.parse_args()
    print(args)

    if args.normalize:
        num_of_norm_vals = len(args.normalize)
        assert num_of_norm_vals == 2 or num_of_norm_vals == 6
        mean = torch.tensor(args.normalize[: num_of_norm_vals // 2])
        std = torch.tensor(args.normalize[(num_of_norm_vals // 2):])
    else:
        mean, std = [None, None]

    test_data_set = LoadNumpyDataset(args.data, mean=mean, std=std)
    test_data_loader = DataLoader(
        test_data_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    n_classes = len(set(test_data_set.labels))

    if args.model == "regression":
        model = Regression(n_classes).cuda()
    else:
        model = CNN(n_classes, args.features == 'packets').cuda()

    initialize_model(model, args.classifier)
    model.eval()

    correct_labels = []
    predicted_labels = []

    with torch.no_grad():
        for test_batch in iter(test_data_loader):
            batch_images = test_batch["image"].cuda(non_blocking=True)
            batch_labels = test_batch["label"].cuda(non_blocking=True)

            out = model(batch_images)
            out_labels = torch.max(out, dim=-1)[1]

            correct_labels.extend(batch_labels)
            predicted_labels.extend(out_labels)

    return confusion_matrix(correct_labels, predicted_labels)


if __name__ == "__main__":
    calculate_confusion_matrix()
