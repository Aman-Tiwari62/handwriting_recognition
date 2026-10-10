
import torch
import torch.nn as nn

from models.cnn import CNNEncoder
from models.rnn import BiLSTM
from models.classifier import CharacterClassifier


class HTRModel(nn.Module):
    def __init__(self, num_classes=78):
        super().__init__()

        self.cnn = CNNEncoder()

        # CNN output: [B, 256, 16, W']
        self.projection = nn.Linear(256 * 16, 256)

        self.rnn = BiLSTM()

        self.classifier = CharacterClassifier(
            input_size=512,
            num_classes=num_classes
        )

    def get_output_lengths(self, input_widths):
        # Two MaxPool2d layers, each with kernel_size=2 and stride=2.
        output_widths = input_widths.clone()
        output_widths = output_widths // 2
        output_widths = output_widths // 2
        return output_widths

    def forward(self, x, input_widths=None):
        # 1. Extract visual features.
        x = self.cnn(x)
        # [B, 256, 16, W']

        # 2. Convert the feature map into a sequence.
        x = x.permute(0, 3, 1, 2)
        # [B, W', 256, 16]

        batch_size, sequence_width, channels, height = x.shape
        x = x.reshape(
            batch_size,
            sequence_width,
            channels * height
        )
        # [B, W', 4096]

        # 3. Project each timestep to 256 features.
        x = self.projection(x)
        # [B, W', 256]

        # 4. Process only valid timesteps when widths are provided.
        if input_widths is not None:
            sequence_lengths = self.get_output_lengths(input_widths)
            x = self.rnn(x, sequence_lengths)
        else:
            x = self.rnn(x)

        # [B, W', 512]

        # 5. Predict character classes at every timestep.
        x = self.classifier(x)
        # [B, W', 78]

        return x