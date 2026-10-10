class Vocabulary:
    def __init__(self):
        # CTC requires a special blank token.
        self.characters = [ "<BLANK>", " ", "!", "'", '"', "#", "(", ")", "*", ",", "-", ".", "/", ":", ";", "?", ]

        # Add lowercase letters
        self.characters += list("abcdefghijklmnopqrstuvwxyz")

        # Add uppercase letters
        self.characters += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # Add digits
        self.characters += list("0123456789")

        self.char_to_index = {
            char: index
            for index, char in enumerate(self.characters)
        }

        self.index_to_char = {
            index: char
            for index, char in enumerate(self.characters)
        }

    def encode(self, text):
        return [
            self.char_to_index[char]
            for char in text
            if char in self.char_to_index
        ]

    def decode(self, indices):
        text = ""

        for index in indices:
            if index == 0:
                # CTC blank
                continue

            text += self.index_to_char[index]

        return text