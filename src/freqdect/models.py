import torch
import numpy as np
from torch.nn.modules.activation import ReLU


def compute_parameter_total(net: torch.nn.Module) -> int:
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


class CNN(torch.nn.Module):
    def __init__(self, classes: int, packets: bool):
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
    def __init__(self, classes:int):
        super().__init__()
        self.linear = torch.nn.Linear(49152, classes)

        # self.activation = torch.nn.Sigmoid()
        self.activation = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.activation(self.linear(x_flat))


class MLP(torch.nn.Module):
    def __init__(self, classes:int):
        super().__init__()

        self.classifier = \
            torch.nn.Sequential(
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
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.activation(self.classifier(x_flat))
