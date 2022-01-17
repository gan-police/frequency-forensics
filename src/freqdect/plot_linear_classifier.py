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
    return parser.parse_args()


def main(args):
    """Plot the weights from the linear classifer defined in the models-module."""
    model = Regression(2)
    model.load_state_dict(torch.load(args.model_path))
    mat = torch.reshape(model.linear.weight.cpu().detach(), [2, 64, 16, 16, 3])
    mat = torch.mean(mat, -1)
    real_weights = generate_frequency_packet_image(mat[0].numpy(), 3)
    fake_weights = generate_frequency_packet_image(mat[1].numpy(), 3)
    plt.imshow(np.concatenate([real_weights, fake_weights], axis=1))
    plt.title("Real and fake class weights side by side.")
    plt.colorbar()
    plt.show()

    plt.imshow(real_weights, cmap=plt.cm.viridis, vmax=0.2)
    plt.title("Real classifier weights")
    plt.colorbar()
    tikz.save("real_classifier_weights.tex")
    plt.show()

    plt.imshow(fake_weights, cmap=plt.cm.viridis, vmax=0.2)
    plt.title("fake classifier weights")
    plt.colorbar()
    tikz.save("fake_classifier_weights.tex")
    plt.show()

    print("stop")


if __name__ == "__main__":
    main(_parse_args())
