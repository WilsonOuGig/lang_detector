import pandas as pd


file = "data/train-00007-of-00041.parquet"  # one of huggingface wiki file
file = "data/train-00007-of-00041.parquet"  # one of huggingface wiki file
# file = "data/train-00005-of-00041.parquet"  # one of huggingface wiki file


file = "parquet/en.train-00028-of-00041.parquet"  # one of huggingface wiki file
file = "parquet/fr.train-00011-of-00017.parquet"



df = pd.read_parquet(file, engine='fastparquet')


print(df.columns)
print(df.head(5))


text = " ".join(df["text"].values.tolist())

with open(file.replace(".parquet", ".txt"), 'wt', encoding='utf-8') as f:
    f.write(text)





