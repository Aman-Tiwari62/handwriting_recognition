from dataset import IAMWordDataset
from preprocessing import transform


dataset = IAMWordDataset(
    csv_file="data/processed/words.csv",
    dataset_root="data/raw/iam_words",
    transform=transform
)


print("Dataset size:", len(dataset))

image, label, text = dataset[0]

print("Image tensor shape:", image.shape)
print("Label:", label)
print("Text:", text)