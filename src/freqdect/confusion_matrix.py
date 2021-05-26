import argparse
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

    n_classes = len(set(test_data_set.labels))

    if args.model == "regression":
        model = Regression(n_classes).cuda()
    else:
        model = CNN(n_classes, args.features == 'packets').cuda()

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

            correct_labels.extend(batch_labels.cpu())
            predicted_labels.extend(out_labels.cpu())

    return confusion_matrix(correct_labels, predicted_labels)


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
    args = parser.parse_args()

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
