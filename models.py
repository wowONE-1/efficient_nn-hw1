from collections import OrderedDict

import torch
from torch import nn


class myCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("pool", nn.MaxPool2d(3, stride=2, padding=1)),
                    ("conv2", nn.Conv2d(32, 64, 5, padding=2, bias=False)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("conv3", nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("conv4", nn.Conv2d(128, 256, 1, padding=0, bias=False)),
                    ("relu4", nn.ReLU(inplace=True)),
                    ("conv5", nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)),
                    ("relu5", nn.ReLU(inplace=True)),
                    ("conv6", nn.Conv2d(256, 512, 1, padding=0, bias=False)),
                    ("relu6", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(512, 256)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(256, 100)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
