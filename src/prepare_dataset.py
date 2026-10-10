from pathlib import Path
import csv


# Change this if your dataset is somewhere else
DATASET_DIR = Path("data/raw/iam_words")

WORDS_FILE = DATASET_DIR / "words.txt"
OUTPUT_FILE = Path("data/processed/words.csv")


def prepare_dataset():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = []

    with open(WORDS_FILE, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Actual IAM format:
            #
            # word_id status graylevel x y width height tag transcription
            #
            # Example:
            # a01-000u-00-01 ok 154 507 766 213 48 NN MOVE

            if len(parts) < 9:
                print("Malformed line:", repr(line))
                print("Parts:", parts)
                continue

            word_id = parts[0]
            status = parts[1]
            transcription = parts[-1]

            # We only want correctly segmented words
            if status != "ok":
                continue

            # Extract directory components
            #
            # a01-000u-00-01
            #   │
            #   ├── a01
            #   └── a01-000u

            parts_id = word_id.split("-")

            writer_id = parts_id[0]
            line_id = "-".join(parts_id[:2])

            image_path = (
                Path("words")
                / writer_id
                / line_id
                / f"{word_id}.png"
            )

            records.append(
                {
                    "image_path": str(image_path),
                    "text": transcription,
                }
            )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "text"]
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Total samples: {len(records)}")


if __name__ == "__main__":
    prepare_dataset()