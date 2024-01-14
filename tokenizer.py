from utils import remove_emojis_and_eol_from_list


class CharTokenizer:
    def __init__(self, text):
        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        chars = remove_emojis_and_eol_from_list(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars) + 1 # last index is  for unknown char
        itos[self.vocab_size - 1] = ' '  # unknown char decoded as space, we don't use it anyway

        self.encode = lambda s: [stoi.get(c, self.vocab_size -1 ) for c in s] # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# import importlib
import tiktoken
class TikTokenizer:
    def __init__(self):
        # analyze tokenzie by titoken
        # print("tiktoken version:", importlib.metadata.version("tiktoken"))
        tokenizer = tiktoken.get_encoding("gpt2")

        self.decode = tokenizer.decode
        self.encode = tokenizer.encode
        self.vocab_size = tokenizer.n_vocab



import re
class WordTokenizer:
    def __init__(self, text):
        # here are all the unique characters that occur in this text
        words = re.split(r'([,.?_!"()\']|--|\s)', text)
        words = [item.strip() for item in words if item.strip()]
        words = sorted(list(set(words)))
        self.vocab_size = len(words)

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text





# tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.n_vocab)
# print(tokenizer.encode("may this is good I"))
# print(tokenizer.encode("I say this is good day and night"))
# print(tokenizer.n_vocab)
#
# ids = [5661, 318, 922]
# print(tokenizer.decode(ids))
# ids = [40, 910, 428, 318, 922, 1110, 290, 1755]
# print(tokenizer.decode(ids))
#
