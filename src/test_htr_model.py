import torch

from models.htr_model import HTRModel


model = HTRModel()

x = torch.randn(4, 1, 64, 279)

input_widths = torch.tensor(
    [113, 247, 279, 146]
)

logits = model(x)

output_lengths = model.get_output_lengths(
    input_widths
)

print("Input shape:", x.shape)
print("Output shape:", logits.shape)

print("Input widths:", input_widths)
print("Output lengths:", output_lengths)