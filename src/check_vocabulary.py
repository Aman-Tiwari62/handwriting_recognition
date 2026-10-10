
import pandas as pd

from vocabulary import Vocabulary


# Load all word transcriptions
df = pd.read_csv("data/processed/words.csv")

vocab = Vocabulary()

supported_chars = set(vocab.char_to_index.keys())
dataset_chars = set("".join(df["text"].astype(str)))

# Find characters missing from the vocabulary
unknown_chars = sorted(dataset_chars - supported_chars)

print("Total word samples:", len(df))
print("Unique characters in dataset:", len(dataset_chars))
print("Characters in vocabulary:", len(supported_chars))

if unknown_chars:
    print("\nUnsupported characters found:")
    for char in unknown_chars:
        print(f"  {repr(char)}  Unicode: U+{ord(char):04X}")
else:
    print("\nAll dataset characters are supported by the vocabulary!")

