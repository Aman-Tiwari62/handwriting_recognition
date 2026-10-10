from torch.utils.data import DataLoader

from dataset import IAMWordDataset
from preprocessing import transform
from collate import collate_fn


def create_loader(csv_path, shuffle=False):
    dataset = IAMWordDataset(
        csv_file=csv_path,
        dataset_root="data/raw/iam_words",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return dataset, loader


train_dataset, train_loader = create_loader(
    "data/processed/train.csv",
    shuffle=True
)

val_dataset, val_loader = create_loader(
    "data/processed/val.csv"
)

test_dataset, test_loader = create_loader(
    "data/processed/test.csv"
)


print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


images, labels, input_widths, label_lengths, texts = next(iter(train_loader))

print("\nTraining batch:")
print("Images:", images.shape)
print("Labels:", labels.shape)
print("Input widths:", input_widths)
print("Label lengths:", label_lengths)
print("Texts:", texts)
