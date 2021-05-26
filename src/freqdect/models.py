from os import PathLike
from typing import Union, BinaryIO, IO

import torch
import numpy as np


class CNN(torch.nn.Module):
    def __init__(self, classes, packets):
        super().__init__()
        self.packets = packets

        if self.packets:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(192, 24, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 6),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, 9),
                torch.nn.ReLU()
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
                torch.nn.ReLU())
            self.linear = torch.nn.Linear(32 * 28 * 28, classes)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x = generate_packet_image_tensor(x)
        if self.packets:
            # batch_size, packets, height, width, channels
            shape = x.shape
            # batch_size, height, width, packets, channels
            x = x.permute([0, 2, 3, 1, 4])
            # batch_size, height, width, packets*channels
            x = x.reshape([shape[0], shape[2], shape[3], shape[1]*shape[4]])
            # batch_size, packets*channels, height, width
        x = x.permute([0, 3, 1, 2])

        out = self.layers(x)
        out = torch.reshape(out, [out.shape[0], -1])
        out = self.linear(out)
        return self.logsoftmax(out)


class Regression(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.linear = torch.nn.Linear(49152, classes)

        # self.activation = torch.nn.Sigmoid()
        self.activation = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.activation(self.linear(x_flat))


def compute_parameter_total(net: torch.nn.Module):
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


def save_model(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)


def initialize_model(model: torch.nn.Module, path):
    model.load_state_dict(torch.load(path))
