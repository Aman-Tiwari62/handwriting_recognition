
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

CSV_PATH = Path("data/processed/words.csv")
DATASET_ROOT = Path("data/raw/iam_words")
OUTPUT_PATH = Path("data/processed/words.csv")

data = pd.read_csv(CSV_PATH)

valid_rows = []
invalid_paths = []

for _, row in data.iterrows():
    image_path = DATASET_ROOT / row["image_path"]

    try:
        with Image.open(image_path) as image:
            image.verify()

        valid_rows.append(row)

    except (OSError, ValueError, UnidentifiedImageError) as error:
        invalid_paths.append(str(image_path))
        print(f"Skipping invalid image: {image_path}")
        print(f"Reason: {error}")

cleaned_data = pd.DataFrame(valid_rows, columns=data.columns)
cleaned_data.to_csv(OUTPUT_PATH, index=False)

print("\nCleaning complete.")
print("Original entries:", len(data))
print("Valid entries:", len(cleaned_data))
print("Invalid entries skipped:", len(invalid_paths))
print("Saved cleaned dataset to:", OUTPUT_PATH)