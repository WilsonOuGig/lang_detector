import torch
import torch.nn as nn
from torch.nn import functional as F

import llm_config
from utils import get_device

device = get_device()


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config):
        n_embd = config['n_embd']
        num_heads = config['n_head']
        head_size = n_embd // num_heads

        block_size = config['block_size']
        dropout_percent = config['dropout_percent']

        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


        self.dropout = nn.Dropout(dropout_percent)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config):
        n_embd = config['n_embd']
        dropout_percent = config['dropout_percent']
        num_heads = config['n_head']

        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_percent)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        dropout_percent = config['dropout_percent']
        n_embd = config['n_embd']

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

    def __init__(self, config):
        n_embd = config['n_embd']
        num_heads = config['n_head']

        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LargeLanguageModel(nn.Module):

    def __init__(self, vocab_size, lang_size):
        config = llm_config.nw_config
        n_embd = config['n_embd']
        self.block_size = config['block_size']
        n_layer = config['n_layer']

        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, lang_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,lang_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # need to include loss for each token to deal with short input
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            #
            # logits = logits[:, -1, :]
            # targets = targets[:, -1]

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def predict(self, idx):
        # idx is (B, T) array of indices in the current context

        # crop idx to the last block_size tokens
        idx_cond = idx[:, -self.block_size:]
        # get the predictions
        logits, loss = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C), only retain last token output

        idx_class = torch.argmax(logits, dim=1)
        prob = torch.softmax(logits, dim=1)[:, idx_class]

        return idx_class, prob
