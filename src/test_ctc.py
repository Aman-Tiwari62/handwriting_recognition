import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import IAMWordDataset
from preprocessing import transform
from collate import collate_fn
from models.htr_model import HTRModel


# -----------------------------
# Dataset
# -----------------------------

dataset = IAMWordDataset(
    csv_file="data/processed/words.csv",
    dataset_root="data/raw/iam_words",
    transform=transform
)


# -----------------------------
# DataLoader
# -----------------------------

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


# Get one real batch
images, labels, input_widths, label_lengths, texts = next(iter(loader))


print("Images shape:", images.shape)
print("Labels:", labels)
print("Input widths:", input_widths)
print("Label lengths:", label_lengths)
print("Texts:", texts)


# -----------------------------
# Model
# -----------------------------

model = HTRModel(num_classes=78)

logits = model(images, input_widths)

print("Logits shape:", logits.shape)


# -----------------------------
# Calculate input lengths
# -----------------------------

input_lengths = model.get_output_lengths(
    input_widths
)

print("CTC input lengths:", input_lengths)


# -----------------------------
# Prepare logits for CTC
# -----------------------------

# Model output:
# [B, T, C]

# CTC expects:
# [T, B, C]

log_probs = torch.log_softmax(
    logits,
    dim=2
)

log_probs = log_probs.permute(1, 0, 2)

print("Log probabilities shape:", log_probs.shape)


# -----------------------------
# CTC Loss
# -----------------------------

ctc_loss = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

loss = ctc_loss(
    log_probs,
    labels,
    input_lengths,
    label_lengths
)

print("CTC Loss:", loss.item())