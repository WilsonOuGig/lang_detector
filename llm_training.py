from os import path

import torch
import torch.nn as nn
from torch.nn import functional as F

import network_config
from network import LargeLanguageModel
from tokenizer import CharTokenizer, ClassificationClassTokenizer
from utils import read_parquet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel?
max_iters = 3000
eval_interval = 100
learning_rate = 3e-4
eval_iters = 20
block_size = network_config.nw_config["block_size"]


# data loading
def get_batch(split, train, val, block_size):
    train_data = train[0]
    val_data = val[0]
    train_lang = train[1]
    val_lang = val[1]

    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    lang = train_lang if split == 'train' else val_lang
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.full(x.shape, lang)

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, block_size)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    torch.manual_seed(1337)

    dataset_files = ["en.train-00028-of-00041.parquet",
                     "fr.train-00011-of-00017.parquet"]  # one of huggingface wiki file
    text_list = {}

    text = ''
    limit_text = 200000  # 10000 #limit it for debug
    for file in dataset_files:
        text = read_parquet(path.join("data/parquet", file))
        if limit_text > 0:
            text = text[:limit_text]
        text_list[file.split('.')[0]] = text  # dict of lang:text

    all_text = [t for t in text_list.values()]
    all_text = " ".join(all_text)

    lang_list = list(text_list.keys())
    lang_list.sort()

    input_tokenizer = CharTokenizer(all_text)
    output_tokenizer = ClassificationClassTokenizer(classes=lang_list)
    lang_size = output_tokenizer.get_class_size()

    encode = input_tokenizer.encode
    vocab_size = input_tokenizer.vocab_size

    # Train and test splits
    train_data = {}
    val_data = {}

    for lang, text in text_list.items():
        lg = torch.tensor(output_tokenizer.encode(lang), dtype=torch.long)

        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be trained, rest val
        train_data[lang] = data[:n], lg
        val_data[lang] = data[n:], lg

    model = LargeLanguageModel(vocab_size, lang_size)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        for lang in lang_list:
            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(m, train_data[lang], val_data[lang])

                x_v, y_v = get_batch('val', train_data[lang], val_data[lang], block_size)
                detect_lang = output_tokenizer.decode(m.predict(x_v[:1, :])[0])

                if lang != detect_lang and iter > 300:
                    print('stop')

                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} language == detected lang: {lang}  == {detect_lang} {'!!!!!!!' if lang == detect_lang else '????????????'} ")
                print("=======================")

            # sample a batch of data
            xb, yb = get_batch('train', train_data[lang], val_data[lang], block_size)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    ## save trained model
    model_save_to_path = "models"
    torch.save(model.state_dict(), path.join(model_save_to_path, "pre-train-llm"))
    input_tokenizer.save_to(model_save_to_path)
    output_tokenizer.save_to(model_save_to_path)


if __name__ == "__main__":
    train()
