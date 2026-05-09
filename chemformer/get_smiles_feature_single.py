import csv
import sys
sys.path.append('/data1/***/codes/Chemformer-main/')
import torch
import pandas as pd
import numpy as np
from molbart.utils.samplers.beam_search_samplers import DecodeSampler
from torch.nn.functional import pad
import molbart.utils.data_utils as util
from molbart.models.transformer_models import BARTModel
from molbart.utils.tokenizers.tokenizers import ChemformerTokenizer

vocabulary_path = util.DEFAULT_VOCAB_PATH
model_path = '/data1/***/codes/Chemformer-main/data/weight/combined-large/step=1000000.ckpt'

# === 非原子 token 表
non_atomic_tokens = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "(", ")", ".", "=", "#", "-", "/", "\\", ":",
    "~", "@", "?", ">", "*", "$",
    "^", "&", "<PAD>", "<MASK>", "<SEP>"
] + [f"<UNUSED_{i}>" for i in range(200)] + [f"%{i}" for i in range(10, 100)]




tokenizer = ChemformerTokenizer(filename=util.DEFAULT_VOCAB_PATH)
vocabulary_size = len(tokenizer)

sampler = DecodeSampler(tokenizer, util.DEFAULT_MAX_SEQ_LEN)

model = BARTModel.load_from_checkpoint(model_path, decode_sampler=sampler, vocabulary_size=vocabulary_size)
model.eval()

model_path = '/data1/***/codes/Chemformer-main/data/weight/combined-large/step=1000000.ckpt'
input_csv_path = "/data1/***/codes/Chemformer-main/data/pdbbindv2020_smiles.csv"
df = pd.read_csv(input_csv_path)

for i in range(df.shape[0]):
    # print('-------------------',i,'----------------------------')
    example_smile = [df.loc[i, "smiles"]]
    n_sample = len(example_smile)
    seq_len = util.DEFAULT_MAX_SEQ_LEN
    pad_token = tokenizer.vocabulary[tokenizer.special_tokens['pad']]
    tokenized_input = tokenizer.encode(example_smile)
    tokens = tokenizer.tokenize(example_smile)[0]
    print(tokens)
    tokens_np = np.array(tokens)

    atom_keep = [token not in non_atomic_tokens for token in tokens]
    atom_drop = [token in non_atomic_tokens for token in tokens]
    print(tokens_np[atom_keep])
    print(tokens_np[atom_drop])

    inputs = []
    masks = []
    for s in tokenized_input:
        mask = torch.zeros(s.shape, dtype=torch.bool)
        inputs.append(pad(s, pad=(0, seq_len - s.shape[0]), value=pad_token))
        masks.append(pad(mask, pad=(0, seq_len - s.shape[0]), value=True))


    batch = {
        "encoder_input": torch.reshape(torch.cat(inputs), (n_sample, seq_len)).T,
        "encoder_pad_mask": torch.reshape(torch.cat(masks), (n_sample, seq_len)).T,
    }

    outputs = model.encode(batch)

    mask = ~batch['encoder_pad_mask'][:, 0]
    embedded = outputs[mask, 0, :]
    atom_feature = embedded[atom_keep]
    print(atom_feature)

    if(atom_feature[0].shape[0] != df.loc[i, "num_atoms"]):
            print(df.loc[i, "pdbid"],atom_feature.shape, 'vs', df.loc[i, "num_atoms"])