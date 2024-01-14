import torch
from torch import tensor

from llm_training import LargeLanguageModel, device
from tokenizer import TikTokenizer

tokenizer = TikTokenizer()
encode = tokenizer.encode
decode = tokenizer.decode
vocab_size = tokenizer.vocab_size


## load trained model
model_save_to = "models/pre-train-llm"
trained_model = LargeLanguageModel(vocab_size)
trained_model.load_state_dict(torch.load(model_save_to))

def generate_text(max_new_tokens):
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = trained_model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(output)


def complete_text(text, max_new_tokens, temperature=None, top_p=None):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for id in encode(text):
        context = torch.cat((context,  tensor([[id]])), dim=1)  # (B, T+1)

    output = trained_model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)[0].tolist()
    return decode(output)


def demo_temperature(temperatures):
    new_token = 30
    print(f"Complete text up to {new_token} tokens:")

    text = "He was best known for his participation"

    for temperature in temperatures:
        print(" ")
        print(f"---------------------- temperature={temperature}---------------------- ")
        for _ in range(10):
            print("========================================")
            print(complete_text(text, new_token, temperature=temperature))


def demo_top_p(top_p_list):
    new_token = 30
    print(f"Complete text up to {new_token} tokens:")

    text = "He was best known for his participation"

    for top_p in top_p_list:
        print(" ")
        print(f"---------------------- top_p={top_p}---------------------- ")
        for _ in range(10):
            print("========================================")
            print(complete_text(text, new_token, top_p=top_p))


if __name__ == "__main__":
    new_token = 30
    # print("Generating text up to 100 tokens:")
    # print(generate_text(new_token))

    # # print("========================================")
    # text = "He was best known for his participation"
    # print(complete_text(text, new_token))

    # demo_temperature([1, 0.4])
    demo_top_p([1, 0.7])