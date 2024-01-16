import json
from os import path

from utils import remove_emojis_and_eol_from_list


class CharTokenizer:
    saved_file_name = "tokenizer.json"

    def __init__(self, text: str = None, saved_path: str = None):
        if text is not None:
            # here are all the unique characters that occur in this text
            chars = sorted(list(set(text)))
            chars = remove_emojis_and_eol_from_list(chars)
        elif saved_path is not None:
            chars = self.load(saved_path)
        else:
            raise ValueError("text or saved_path is needed to initialize CharTokenizer")

        self.chars = chars

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # last index is  for unknown char
        itos[self.vocab_size - 1] = ' '  # unknown char decoded as space, we don't use it anyway

        self.encode = lambda s: [stoi.get(c, self.vocab_size - 1) for c in
                                 s]  # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    def save_to(self, save_to_path):
        with open(path.join(save_to_path, self.saved_file_name), 'wt') as f:
            json.dump(self.chars, f)

    def load(self, save_to_path):
        with open(path.join(save_to_path, self.saved_file_name), 'rt') as f:
            return json.load(f)


class ClassificationClassTokenizer:
    saved_file_name = "classes.json"

    def __init__(self, classes: list = None, saved_path: str = None):
        if classes is not None:
            self.classes = classes
        elif saved_path is not None:
            self.classes = self.load(saved_path)
        else:
            raise ValueError("Either classes or saved_path is needed to initialize CharTokenizer")

        self.classes2i = {cls: i for i, cls in enumerate(self.classes)}
        self.encode = lambda cls: self.classes2i[cls]
        self.decode = lambda i: self.classes[i]

    def save_to(self, save_to_path):
        with open(path.join(save_to_path, self.saved_file_name), 'wt') as f:
            json.dump(self.classes, f)

    def load(self, save_to_path):
        with open(path.join(save_to_path, self.saved_file_name), 'rt') as f:
            return json.load(f)

    def get_class_size(self):
        return len(self.classes)
