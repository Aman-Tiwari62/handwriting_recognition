import torch

from models.rnn import BiLSTM


model = BiLSTM()

# Output of our CNN + projection
x = torch.randn(4, 69, 256)

output = model(x)

print("Input shape:", x.shape)
print("BiLSTM output shape:", output.shape)