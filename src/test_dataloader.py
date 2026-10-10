from torch.utils.data import DataLoader

from dataset import IAMWordDataset
from preprocessing import transform
from collate import collate_fn


dataset = IAMWordDataset(
    csv_file="data/processed/words.csv",
    dataset_root="data/raw/iam_words",
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


batch = next(iter(loader))

images, labels, input_widths, label_lengths, texts = batch


print("Images shape:", images.shape)
print("Labels:", labels)
print("Input widths:", input_widths)
print("Label lengths:", label_lengths)
print("Texts:", texts)