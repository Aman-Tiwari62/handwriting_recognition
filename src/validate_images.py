
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

CSV_PATH = Path("data/processed/words.csv")
DATASET_ROOT = Path("data/raw/iam_words")

data = pd.read_csv(CSV_PATH)

bad_paths = []

for index, row in data.iterrows():
    image_path = DATASET_ROOT / row["image_path"]

    try:
        with Image.open(image_path) as image:
            image.verify()

    except (OSError, ValueError, UnidentifiedImageError) as error:
        bad_paths.append((index, str(image_path), str(error)))

print("Total dataset entries:", len(data))
print("Invalid or unreadable images:", len(bad_paths))

for index, path, error in bad_paths:
    print(f"Row {index}: {path}")
    print(f"  Error: {error}")

if not bad_paths:
    print("All images passed the initial validation.")