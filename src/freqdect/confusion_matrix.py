import argparse
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .models import CNN, Regression, initialize_model
from .data_loader import LoadNumpyDataset


def calculate_confusion_matrix(args):
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

    if args.model == "regression":
        model = Regression(args.nclasses).cuda()
    else:
        model = CNN(args.nclasses, args.features == 'packets').cuda()

    initialize_model(model, args.classifier_path)
    model.eval()

    correct_labels = []
    predicted_labels = []

    with torch.no_grad():
        for test_batch in iter(test_data_loader):
            batch_images = test_batch["image"].cuda(non_blocking=True)
            batch_labels = test_batch["label"]

            out = model(batch_images)
            out_labels = torch.max(out, dim=-1)[1]

            if args.nclasses == 2:
                batch_labels[batch_labels > 0] = 1

            correct_labels.extend(batch_labels.cpu())
            predicted_labels.extend(out_labels.cpu())

    return confusion_matrix(correct_labels, predicted_labels)


def confusion_matrix_generalized(args):
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

    if args.model == "regression":
        model = Regression(args.nclasses).cuda()
    else:
        model = CNN(args.nclasses, args.features == 'packets').cuda()

    initialize_model(model, args.classifier_path)
    model.eval()

    predicted_dict = defaultdict(list)

    label_names = np.array(["Original", "CramerGAN", "MMDGAN", "ProGAN", "SNGAN"])

    with torch.no_grad():
        for test_batch in iter(test_data_loader):
            batch_images = test_batch["image"].cuda(non_blocking=True)
            batch_labels = test_batch["label"].cpu()

            batch_names = label_names[batch_labels]

            out = model(batch_images)
            out_labels = torch.max(out, dim=-1)[1].cpu()

            for k, v in zip(batch_names, out_labels):
                predicted_dict[k].append(v)

    matrix = np.zeros((len(label_names), args.nclasses), dtype=int)

    for label_idx, label in enumerate(label_names):
        predicted_labels = np.array(predicted_dict[label])

        for class_idx in range(args.nclasses):
            matrix[label_idx, class_idx] = len(predicted_labels[predicted_labels == class_idx])

    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the confusion matrix")
    parser.add_argument(
        "--classifier-path",
        type=str,
        help="path to classifier model file"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="path of folder containing the test data"
    )
    parser.add_argument(
        "--model",
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="plot the confusion matrix and store the plot as png"
    )
    parser.add_argument(
        "--nclasses", type=int, default=2, help="number of classes (default: 2)"
    )
    parser.add_argument(
        "--generalized", action="store_true"
    )
    args = parser.parse_args()

    if args.generalized:
        matrix = confusion_matrix_generalized(args)
        print(matrix)
    else:
        confusion_matrix = calculate_confusion_matrix(args)

        print('accuracy: ', np.trace(confusion_matrix)/confusion_matrix.sum())

        label_names = ["Original", "CramerGAN", "MMDGAN", "ProGAN", "SNGAN"]

        diag = np.diag(confusion_matrix)

        worst_index = np.argmin(diag)
        best_index = np.argmax(diag)
        print(f"worst index: {worst_index} ({label_names[worst_index]}) with an accuracy of {diag[worst_index]/confusion_matrix[worst_index].sum()*100:.2f}%")
        print(f"best index: {best_index} ({label_names[best_index]}) with an accuracy of {diag[best_index]/confusion_matrix[best_index].sum()*100:.2f}%")

        print(confusion_matrix)

        if args.plot:
            import matplotlib.pyplot as plt

            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=label_names)
            disp.plot()
            plt.savefig('confusion_matrix.png')
            plt.show()
