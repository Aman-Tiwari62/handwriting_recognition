from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from vocabulary import Vocabulary


class IAMWordDataset(Dataset):

    def __init__(
        self,
        csv_file,
        dataset_root,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)

        self.dataset_root = Path(dataset_root)

        self.transform = transform

        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.dataset_root / row["image_path"]

        text = row["text"]

        # Load image
        image = Image.open(image_path).convert("L")

        # Apply preprocessing
        if self.transform:
            image = self.transform(image)

        # Convert text to integer labels
        label = self.vocab.encode(text)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label, text