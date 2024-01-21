# cut text in each file into 256 chars (estimate maximum context window needed, truncate it whatever the real context wiindow during training.). Save as json (list) to avoid handling the return char

import glob
import json
import os
import random
import pandas as pd

max_context = 256
max_record = 120*1000
qa_record = 10*1000

for file in glob.glob("parquet/*.parquet"):
    text_file = file.replace(".parquet", "-processed.json").replace("parquet", 'processed')
    if os.path.exists(text_file):
        print(f'file={text_file} is already splitted.')
        continue

    print(f'process {file=} to as dataset.')

    df = pd.read_parquet(file, engine='fastparquet')
    text = " ".join(df["text"].values.tolist())

    records = []
    for i in range(0, len(text) - max_context, max_context):
        rec = text[i: i + max_context]
        records.append(rec)
        if len(records) == max_record:
            break

    with open(text_file, 'wt', encoding='utf-8') as f:
        json.dump(records, f)



#split 10% as QA file. Randomly shuffle the list and get last 10% as qa
random_seed = 42
random.seed(random_seed) # for shuffle dataset order
file_list = {}
for file in glob.glob("processed/*-processed.json"):
    qa_file = file.replace("-processed.", "-qa.")
    if os.path.exists(qa_file):
        print(f'{file=} is already splitted.')
        continue

    print(f'split {file=} to train/val and qa file.')
    with open(file, 'rt') as f:
        text = json.load(f)
    lang = file.split('.')[0].split("/")[-1]
    random.shuffle(text)
    n = int(0.9 * len(text))  # second 10% will be  val
    train_val = text[:n]
    qa = text[n:]

    with open(file, 'wt') as f:
        text = json.dump(train_val, f)

    with open(qa_file, 'wt') as f:
        text = json.dump(qa, f)





