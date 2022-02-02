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
            total += np.prod(p.shape)
    return total


class CNN(torch.nn.Module):
    """CNN models used for packet or pixel classification."""

    def __init__(self, classes: int, feature: str = "image"):
        """Create a convolutional neural network (CNN) model.

        Args:
            classes (int): The number of classes or sources to classify.
            feature (str)): A string which tells us the input feature
                we are using.
        """
        super().__init__()
        self.feature = feature

        if feature == "packets":
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(192, 24, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 6),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 9),
                torch.nn.ReLU(),
            )
            self.linear = torch.nn.Linear(24, classes)
        elif feature == "all-packets" or feature == "all-packets-fourier":
            if feature == "all-packets-fourier":
                self.scale1 = torch.nn.Sequential(
                    torch.nn.Conv2d(6, 8, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AvgPool2d(2, 2),
                )
            else:
                self.scale1 = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 8, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AvgPool2d(2, 2),
                )
            self.scale2 = torch.nn.Sequential(
                torch.nn.Conv2d(20, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
            )
            self.scale3 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2, 2),
            )
            self.scale4 = torch.nn.Sequential(
                torch.nn.Conv2d(224, 32, 3, 1, padding=1), torch.nn.ReLU()
            )
            self.linear = torch.nn.Linear(32 * 16 * 16, classes)
        else:
            # assume an 128x128x3 image input.
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3),
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

    def forward(self, x) -> torch.Tensor:
        """Compute the CNN forward pass.

        Args:
            x (torch.Tensor or dict): An input image of shape
                [batch_size, packets, height, width, channels]
                for packet inputs and
                [batch_size, height, width, channels]
                else.

        Returns:
            torch.tensor: A logsoftmax scaled output of shape
                [batch_size, classes].

        """
        # x = generate_packet_image_tensor(x)
        if self.feature == "packets":
            # batch_size, packets, height, width, channels
            shape = x.shape
            # batch_size, height, width, packets, channels
            x = x.permute([0, 2, 3, 1, 4])
            # batch_size, height, width, packets*channels
            to_net = x.reshape([shape[0], shape[2], shape[3], shape[1] * shape[4]])
            # batch_size, packets*channels, height, width
        elif self.feature == "all-packets":
            to_net = x["raw"]
        elif self.feature == "all-packets-fourier":
            to_net = torch.cat([x["raw"], x["fourier"]], dim=-1)
        else:
            to_net = x

        to_net = to_net.permute([0, 3, 1, 2])

        if self.feature == "all-packets" or self.feature == "all-packets-fourier":
            res = self.scale1(to_net)
            packets = [
                torch.reshape(
                    x[key].permute([0, 2, 3, 1, 4]),
                    [x[key].shape[0], x[key].shape[2], x[key].shape[3], -1],
                ).permute(0, 3, 1, 2)
                for key in ["packets1", "packets2", "packets3"]
            ]
            # shape: batch_size, packet_channels, height, widht, color_channels
            # cat along channel dim1.
            to_net = torch.cat([packets[0], res], dim=1)
            res = self.scale2(to_net)
            to_net = torch.cat([packets[1], res], dim=1)
            res = self.scale3(to_net)
            to_net = torch.cat([packets[2], res], dim=1)
            out = self.scale4(to_net)
            out = torch.reshape(out, [out.shape[0], -1])
            out = self.linear(out)
        else:
            out = self.layers(to_net)
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

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Compute the regression forward pass.

        Args:
            x (torch.tensor): An input tensor of shape
                [batch_size, ...]

        Returns:
            torch.tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.logsoftmax(self.linear(x_flat))


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
