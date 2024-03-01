import os
import pandas as pd
import tiktoken
import numpy as np

input_file_path = 'data/vibhanshu/cleaned_data.txt'
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode(train_data)
val_ids = enc.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('data/vibhanshu/train.bin')
val_ids.tofile('data/vibhanshu/val.bin')

