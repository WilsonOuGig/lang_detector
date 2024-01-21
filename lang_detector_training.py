import json
import time
from os import path

from random import shuffle
import random
import torch
import glob

import llm_config
from lang_detector_inference import LanguageDetector
from llm_network import LargeLanguageModel
from tokenizer import CharTokenizer, ClassificationClassTokenizer
from utils import read_parquet, get_device

device = get_device()

load_from_pretrained = True


# hyperparameters
batch_size = 20 #16  # how many independent sequences will we process in parallel? for one lang.  real batch will be batch_size*lang_size
max_iters = 5000 #1000 # 3000
eval_interval = 150 #100
learning_rate = 3e-4
learning_rate = 1e-3

eval_iters = 8
block_size = llm_config.nw_config["block_size"]


# data loading
def get_batch( input_data, block_size):
    # generate a small batch of data of inputs x and targets y
    x_list = []  # for each language
    y_list = []  # for each language
    for lang, data in input_data.items():
        data_t = data[0]
        lang_t = data[1]
        ix = torch.randint(len(data_t), (batch_size,))
        x = torch.stack([data_t[i][:block_size] for i in ix]) # truncate to block_size
        y = torch.full(x.shape, lang_t)
        x_list.append(x)
        y_list.append(y)

    all_x =  torch.stack(x_list).view(-1, x_list[0].shape[-1])
    all_y = torch.stack(y_list).view(-1, x_list[0].shape[-1])
    all_x, all_y = all_x.to(device), all_y.to(device)
    return all_x, all_y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for data, split in zip([train_data, val_data], ['train', 'val']):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    torch.manual_seed(1337)
    torch.manual_seed(1339)
    torch.manual_seed(1341)
    torch.manual_seed(1343)
    torch.manual_seed(1343999)


    file_list = {}
    for file in glob.glob("data/processed/*.json"):
        text = json.load(open(file, 'rt'))
        lang = file.split('.')[0].split("/")[-1]
        file_list[lang] = text  # dict of lang:list of text

    all_text = [r for t in file_list.values() for r in t]
    all_text = " ".join(all_text)

    lang_list = list(file_list.keys())
    lang_list.sort()

    input_tokenizer = CharTokenizer(all_text)
    output_tokenizer = ClassificationClassTokenizer(classes=lang_list)
    lang_size = output_tokenizer.get_class_size()

    encode = input_tokenizer.encode
    vocab_size = input_tokenizer.vocab_size

    # Train and test splits
    train_data = {}
    val_data = {}
    qa_data = {}

    random_seed = 42
    random.seed(random_seed) # for shuffle dataset order

    for lang, text_list in file_list.items():
        lg = torch.tensor(output_tokenizer.encode(lang), dtype=torch.long)

        data = [torch.tensor(encode(text), dtype=torch.long) for text in text_list]
        shuffle(data)

        n = int(0.8 * len(data))  # first 80% will be trained
        n1 = int(0.9 * len(data))  # second 10% will be  val
        train_data[lang] = data[:n], lg
        val_data[lang] = data[n:n1], lg
        qa_data[lang] = data[n1:], lg

    if load_from_pretrained:
        model = LanguageDetector().trained_model
    else:
        model = LargeLanguageModel(vocab_size, lang_size)

    model.train() # set as training mode.

    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()
    start_time_eval_interval = time.time()

    for iter in range(max_iters):

        # for lang in lang_list:
            # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:

            end_time_eval_interval = time.time()


            losses = estimate_loss(m, train_data, val_data)

            x_v, y_v = get_batch(val_data, block_size)
            detect_lang= output_tokenizer.decode(m.predict(x_v[:1, :])[0][0])

            if lang != detect_lang and iter > 300:
                print('stop')

            print(
                f"step {iter}: time={(end_time_eval_interval-start_time_eval_interval) :.3f}  train loss {losses['train']:.4f}, val loss {losses['val']:.4f} language == detected lang: {lang}  == {detect_lang} {'!!!!!!!' if lang == detect_lang else '????????????'} ")
            start_time_eval_interval = time.time()
            print("=======================")

        # sample a batch of data
        xb, yb = get_batch(train_data, block_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    end_time = time.time()

    ## save trained model
    model_save_to_path = "models"
    torch.save(model.state_dict(), path.join(model_save_to_path, "llm_model"))
    input_tokenizer.save_to(model_save_to_path)
    output_tokenizer.save_to(model_save_to_path)

    return end_time - start_time


if __name__ == "__main__":


    duration = train()


    print(f"training successful time={duration}")
