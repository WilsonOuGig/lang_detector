import pandas as pd

import glob
import os

file = "data/train-00007-of-00041.parquet"  # one of huggingface wiki file
file = "data/train-00007-of-00041.parquet"  # one of huggingface wiki file
# file = "data/train-00005-of-00041.parquet"  # one of huggingface wiki file


file = "parquet/en.train-00028-of-00041.parquet"  # one of huggingface wiki file
file = "parquet/fr.train-00011-of-00017.parquet"



for file in glob.glob("parquet/*.parquet"):
    text_file = file.replace(".parquet", ".txt")
    if os.path.exists(text_file):
        continue

    df = pd.read_parquet(file, engine='fastparquet')
    text = " ".join(df["text"].values.tolist())
    with open(text_file, 'wt', encoding='utf-8') as f:
        f.write(text)





