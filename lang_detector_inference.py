import torch

from llm_training import LargeLanguageModel
from tokenizer import CharTokenizer, ClassificationClassTokenizer

from time import time


device = 'cuda' if torch.cuda.is_available() else  (torch.device("mps") if torch.backends.mps.is_available() else 'cpu')

class LanguageDetector:
    def __init__(self):
        model_save_to = "models/llm-model"
        model_params = torch.load(model_save_to)
        vocab_size, _ = model_params.get('token_embedding_table.weight').shape
        self.block_size = model_params.get('position_embedding_table.weight').shape[0]
        lang_size =  model_params.get('lm_head.bias').shape[0]
        self.trained_model = LargeLanguageModel(vocab_size, lang_size)
        self.trained_model.load_state_dict(model_params)
        self.input_tokenizer = CharTokenizer(saved_path="models")
        self.output_tokenizer = ClassificationClassTokenizer(saved_path="models")



    def detect(self, text:str):
        text = text[:min(len(text), self.block_size)] # limit to context lenght

        # generate from the model
        input = torch.tensor(self.input_tokenizer.encode(text), dtype=torch.long, device=device)
        input = input.view(1, -1)

        output = self.trained_model.predict(input).tolist()

        return self.output_tokenizer.decode(output[0])



if __name__ == "__main__":

    lang_detector = LanguageDetector()

    while True:
        text =  input('Input: ')
        if not text.strip():
            continue

        t1 = time()
        lang = lang_detector.detect(text)

        print(f"'{text}'  '{lang=}', time={(time() - t1) :.3f}")