import torch

from models.classifier import CharacterClassifier


model = CharacterClassifier()

# Output from BiLSTM
x = torch.randn(4, 69, 512)

logits = model(x)

print("Input shape:", x.shape)
print("Logits shape:", logits.shape)