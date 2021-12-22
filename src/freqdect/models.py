"""This module contains code for deepfake detection models."""
import numpy as np
import torch


def compute_parameter_total(net: torch.nn.Module) -> int:
    """Compute the parameter total of the input net.

    Args:
        net (torch.nn.Module): The model containing the
            parameters to count.

    Returns:
        int: The parameter total.
    """
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)  # type: ignore
    return total


class CNN(torch.nn.Module):
    """CNN models used for packet or pixel classification."""

    def __init__(self, classes: int, packets: bool):
        """Create a convolutional neural network (CNN) model.

        Args:
            classes (int): The number of classes or sources to classify.
            packets (bool): If true we expect wavelet packets as input.
        """
        super().__init__()
        self.packets = packets

        if self.packets:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(192, 24, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 6),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 9),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(24, classes)
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 8, 3),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
                torch.nn.Conv2d(8, 16, 3),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
                torch.nn.Conv2d(16, 32, 3),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(32 * 28 * 28, classes)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the CNN forward pass.

        Args:
            x (torch.Tensor): An input image of shape
                [batch_size, packets, height, width, channels]
                for packet inputs and
                [batch_size, height, width, channels]
                else.

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        # x = generate_packet_image_tensor(x)
        if self.packets:
            # batch_size, packets, height, width, channels
            shape = x.shape
            # batch_size, height, width, packets, channels
            x = x.permute([0, 2, 3, 1, 4])
            # batch_size, height, width, packets*channels
            x = x.reshape([shape[0], shape[2], shape[3], shape[1] * shape[4]])
            # batch_size, packets*channels, height, width
        x = x.permute([0, 3, 1, 2])

        out = self.layers(x)
        out = torch.reshape(out, [out.shape[0], -1])
        out = self.linear(out)
        return self.logsoftmax(out)


class Regression(torch.nn.Module):
    """A shallow linear-regression model."""

    def __init__(self, classes: int):
        """Create the regression model.

        Args:
            classes (int): The number of classes or sources to classify.
        """
        super().__init__()
        self.linear = torch.nn.Linear(49152, classes)

        # self.activation = torch.nn.Sigmoid()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the regression forward pass.

        Args:
            x (torch.Tensor): An input tensor of shape
                [batch_size, ...]

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.logsoftmax(self.linear(x_flat))


class MLP(torch.nn.Module):
    """Create a more involved Multi Layer Perceptron.

    Args:
        torch ([type]): [description]

    - We did not end up using ths MLP in the paper -.
    """

    def __init__(self, classes: int):
        """Create the MLP.

        Args:
            classes (int): The number of classes or sources to classify.
        """
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(49152, 2048, bias=True),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(2048, classes, bias=True),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(p=0.5, inplace=False),
            # torch.nn.Linear(1024, classes, bias=True),
        )

        # self.activation = torch.nn.Sigmoid()
        self.activation = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """Compute the mlp forward pass.

        Args:
            x (torch.Tensor): An input tensor of shape
                [batch_size, ...]

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.activation(self.classifier(x_flat))


def save_model(model: torch.nn.Module, path):
    """Save the state dict of the model to the specified path.

    Args:
        model (torch.nn.Module): model to store
        path: file path of the storage file
    """
    torch.save(model.state_dict(), path)


def initialize_model(model: torch.nn.Module, path):
    """Initialize the given model from a stored state dict file.

    Args:
        model (torch.nn.Module): model to initialize
        path: file path of the storage file
    """
    model.load_state_dict(torch.load(path))
