
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_CSV = Path("data/processed/words.csv")
OUTPUT_DIR = Path("data/processed")

SEED = 42


def main():
    df = pd.read_csv(INPUT_CSV)

    # Word IDs look like a01-000u-00-00.
    # The first component identifies the writer.
    df["writer_id"] = df["image_path"].apply(
        lambda path: Path(path).stem.split("-")[0]
    )

    writers = df["writer_id"].unique()

    # Split writers: 80% train, 20% temporary.
    train_writers, temp_writers = train_test_split(
        writers,
        test_size=0.20,
        random_state=SEED
    )

    # Split the temporary writers equally into validation and test.
    val_writers, test_writers = train_test_split(
        temp_writers,
        test_size=0.50,
        random_state=SEED
    )

    train_df = df[df["writer_id"].isin(train_writers)].copy()
    val_df = df[df["writer_id"].isin(val_writers)].copy()
    test_df = df[df["writer_id"].isin(test_writers)].copy()

    # Remove the helper column from saved CSVs.
    for name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        split_df = split_df.drop(columns=["writer_id"])
        output_path = OUTPUT_DIR / f"{name}.csv"
        split_df.to_csv(output_path, index=False)
        print(f"Saved {output_path}: {len(split_df)} samples")

    # Confirm there is no writer overlap.
    train_set = set(train_writers)
    val_set = set(val_writers)
    test_set = set(test_writers)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    print("\nWriter split verification passed.")
    print("Training writers:", len(train_set))
    print("Validation writers:", len(val_set))
    print("Test writers:", len(test_set))


if __name__ == "__main__":
    main()
