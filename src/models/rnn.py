
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

    def forward(self, x, sequence_lengths=None):
        # If lengths are provided, tell the LSTM which timesteps
        # contain real image features and which are padding.
        if sequence_lengths is not None:
            packed_x = pack_padded_sequence(
                x,
                sequence_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

            packed_output, _ = self.lstm(packed_x)

            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1)
            )

            return output

        # Allows the model to run without lengths during simple
        # shape tests.
        output, _ = self.lstm(x)
        return output