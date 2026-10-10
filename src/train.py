
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import IAMWordDataset
from preprocessing import transform
from collate import collate_fn
from models.htr_model import HTRModel


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 1

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
DATASET_ROOT = "data/raw/iam_words"
CHECKPOINT_DIR = Path("checkpoints")

DEVICE = torch.device("cpu")


# -----------------------------
# Data loaders
# -----------------------------
def create_loader(csv_path, shuffle=False):
    dataset = IAMWordDataset(
        csv_file=csv_path,
        dataset_root=DATASET_ROOT,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0
    )


# -----------------------------
# One training or validation pass
# -----------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(loader, start=1):
        images, labels, input_widths, label_lengths, texts = batch

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        input_lengths = model.get_output_lengths(input_widths)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images, input_widths)

            log_probs = torch.log_softmax(logits, dim=2)
            log_probs = log_probs.permute(1, 0, 2)

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            if training:
                loss.backward()

                # Prevent excessively large gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_index % 100 == 0:
            phase = "Train" if training else "Validation"
            print(
                f"{phase} batch {batch_index}/{len(loader)} "
                f"| batch loss: {loss.item():.4f}"
            )

    return total_loss / total_samples


# -----------------------------
# Main training process
# -----------------------------
def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = create_loader(TRAIN_CSV, shuffle=True)
    val_loader = create_loader(VAL_CSV)

    model = HTRModel(num_classes=78).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    print("Device:", DEVICE)
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion
        )

        print(f"Training loss:   {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "num_classes": 78,
            "characters": model.classifier.linear.out_features
        }

        checkpoint_path = CHECKPOINT_DIR / "latest_model.pt"
        torch.save(checkpoint, checkpoint_path)

        print("Saved checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()