import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            # --------------------------------
            # Block 1
            # --------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # --------------------------------
            # Block 2
            # --------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # --------------------------------
            # Block 3
            # --------------------------------

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

        )

    def forward(self, x):

        return self.cnn(x)