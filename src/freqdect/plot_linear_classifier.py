"""Code to visualize a linear classifier trained on wavelet packets."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch

from .models import Regression
from .plot_mean_packets import generate_frequency_packet_image


def _parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Plot the weights of our linear regressor."
    )
    parser.add_argument(
        "model_path", type=str, help="path to the linear model to plot."
    )
    parser.add_argument(
        "--classes", type=int, help="number of classes in the classifier.",
        default=2
    )
    return parser.parse_args()


def main(args):
    """Plot the weights from the linear classifer defined in the models-module."""
    model = Regression(args.classes)
    model.load_state_dict(torch.load(args.model_path))
    mat = torch.reshape(model.linear.weight.cpu().detach(), [args.classes, 64, 16, 16, 3])
    mat = torch.mean(mat, -1)
    real_weights = generate_frequency_packet_image(mat[0].numpy(), 3)
    fake_weights =  []
    for c in range(1, args.classes):
        fake_weights.append(generate_frequency_packet_image(mat[c].numpy(), 3))

    cat = real_weights
    for fake in fake_weights:
        cat = np.concatenate([cat, fake], axis=1)

    plt.imshow(cat)
    plt.title("Real and fake class weights side by side.")
    plt.colorbar()
    plt.axis("off")
    plt.show()

    plt.imshow(real_weights, cmap=plt.cm.viridis, vmax=0.2, vmin=-0.2)
    plt.title("Real classifier weights")
    plt.colorbar()
    plt.axis("off")
    tikz.save("real_classifier_weights.tex")
    plt.show()

    for c, fake in enumerate(fake_weights):
        plt.imshow(fake, cmap=plt.cm.viridis, vmax=0.2, vmin=-0.2)
        plt.title(f"fake classifier {c+1} weights")
        plt.colorbar()
        tikz.save(f"fake_classifier_weights_{c+1}.tex")
        plt.axis("off")
        plt.show()

    print("stop")


if __name__ == "__main__":
    main(_parse_args())
