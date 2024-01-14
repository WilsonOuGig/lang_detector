import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from tokenizer import CharTokenizer, WordTokenizer, TikTokenizer

from os import path

from utils import read_parquet

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout_percent = 0.0
# ------------


# --  my settings
max_iters = 3000
block_size = 60 # 60 fragment with 60 chars as input
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 32
n_head = 4
n_layer = 6
dropout_percent = 0.01



# data loading
def get_batch(split, train, val):
    train_data = train[0]
    val_data = val[0]
    train_lang = train[1]
    val_lang = val[1]

    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    lang = train_lang if split == 'train' else val_lang
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
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
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout_percent)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_percent)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_percent),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class LargeLanguageModel(nn.Module):

    def __init__(self, vocab_size, lang_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, lang_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,lang_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # logits = logits.view(B*T, C)
            # targets = targets.view(B*T)
            #
            logits = logits[:, -1, :]
            targets =targets[:, -1]

            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def sample(self, logits, temperature:float=None, top_p:float=None):
        if temperature is not None:
            logits = logits / temperature  # when temperature is higher, logits is lower, then soften the distribution in the softmax, create higher entropy (more random)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

        elif top_p is not None:

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sort the probabilities and indices in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            # Calculate cumulative probabilities
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find the nucleus indices
            nucleus_indices = torch.nonzero(cum_probs <= top_p, as_tuple=False)

            if nucleus_indices.numel() == 0:
                # If the entire vocabulary is within the nucleus, use the last index
                nucleus_indices = torch.tensor([[cum_probs.numel() - 1]])

            # Sample from the nucleus
            probs = sorted_probs[:, nucleus_indices[:, 0]]

        else:
            probs = F.softmax(logits, dim=-1)  # (B, C)

        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        return idx_next


    def predict(self, idx, temperature=1, top_p = 1):
        max_new_tokens = 1

        # idx is (B, T) array of indices in the current context

        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C), only retain last token output

        idx_next = max_index = torch.argmax(logits[0])

        return idx_next


def train():
    torch.manual_seed(1337)




    dataset_files = [ "en.train-00028-of-00041.parquet", "fr.train-00011-of-00017.parquet"  ] # one of huggingface wiki file
    text_list = {}


    limit_text = 100000 #10000 #limit it for debug
    for file in dataset_files:
        text =  read_parquet(path.join("data/parquet",file))
        if limit_text> 0:
            text = text[:limit_text]
        text_list[file.split('.')[0]] = text # dict of lang:text

    all_text = [ t[0] for t in text_list]
    all_text = " ".join(all_text)

    tokenizer = CharTokenizer(text)

    # tokenizer = WordTokenizer(text)
    # tokenizer = TikTokenizer()

    encode = tokenizer.encode
    vocab_size = tokenizer.vocab_size

    # Train and test splits
    train_data={}
    val_data={}

    lang_list = list(text_list.keys())
    lang_list.sort()
    lang_size = len(lang_list)


    def encode_lang(lang):
        for i, l in enumerate(lang_list):
            if l == lang:
                return i;

    def decode_lang(i):
        if i< len(lang_list) and i>=0:
            return lang_list[i]
        return "unknown"

    for lang, text in text_list.items():
        lg =  torch.tensor(encode_lang(lang), dtype=torch.long)

        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be trained, rest val
        train_data[lang] = data[:n] , lg
        val_data[lang] = data[n:],  lg

    model = LargeLanguageModel(vocab_size, lang_size)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):


        for lang in lang_list:
            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(m, train_data[lang], val_data[lang])

                # context = torch.zeros((1, 1), dtype=torch.long, device=device)

                x_v, y_v = get_batch('val', train_data[lang], val_data[lang])
                detect_lang = decode_lang(m.predict(x_v[:1, :]))

                if lang != detect_lang and iter>300:
                    print('stop')

                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} language == detected lang: {lang}  == {detect_lang} {'!!!!!!!' if lang  == detect_lang else '????????????'} ")
                print("=======================")


            # sample a batch of data
            xb, yb = get_batch('train', train_data[lang], val_data[lang])

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


    ## save trained model
    model_save_to = "models/pre-train-llm"
    torch.save(model.state_dict(), model_save_to)


if __name__ == "__main__":
    train()
