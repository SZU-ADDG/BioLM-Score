import ast
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from molbart.utils.samplers.beam_search_samplers import DecodeSampler
from torch.nn.functional import pad
import molbart.utils.data_utils as util
from molbart.models.transformer_models import BARTModel
from molbart.utils.tokenizers.tokenizers import ChemformerTokenizer

# === 参数 ===
vocabulary_path = util.DEFAULT_VOCAB_PATH
model_path = '/data1/***/codes/Chemformer-main/data/weight/combined-large/step=1000000.ckpt'
input_csv_path = "/data1/***/codes/Chemformer-main/data/ligand_embeddings2/pdbbindv2020_smiles.csv"
output_dir = "/data1/***/codes/Chemformer-main/data/ligand_embeddings2/pdbbindv2020"
log_file_path = './data/failed_get_ligand_feature.log'
batch_size = 32
seq_len = util.DEFAULT_MAX_SEQ_LEN

# === 非原子 token 表
non_atomic_tokens = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "(", ")", ".", "=", "#", "-", "/", "\\", ":",
    "~", "@", "?", ">", "*", "$",
    "^", "&", "<PAD>", "<MASK>", "<SEP>"
] + [f"<UNUSED_{i}>" for i in range(200)] + [f"%{i}" for i in range(10, 100)]

# === 初始化 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = ChemformerTokenizer(filename=vocabulary_path)
vocabulary_size = len(tokenizer)
sampler = DecodeSampler(tokenizer, util.DEFAULT_MAX_SEQ_LEN)


model = BARTModel.load_from_checkpoint(model_path, decode_sampler=sampler, vocabulary_size=vocabulary_size)
model.to(device)
model.eval()

df = pd.read_csv(input_csv_path)
os.makedirs(output_dir, exist_ok=True)

# === 批处理 + 记录错误 ===
with torch.no_grad(), open(log_file_path, 'w') as log_file:
    for i in tqdm(range(0, len(df), batch_size), desc="Generating atom features"):
        batch_df = df.iloc[i:i+batch_size]
        smiles_list = batch_df["smiles"].tolist()
        pdbid_list = batch_df["pdbid"].tolist()
        num_atoms_list = batch_df["num_atoms"].tolist()
        map_order_list = batch_df["map_order"].apply(ast.literal_eval).tolist()

        # print(map_order_list)

        # Tokenize
        tokenized_input = tokenizer.encode(smiles_list)
        tokens_list = tokenizer.tokenize(smiles_list)

        inputs, masks, atom_keeps = [], [], []

        #生成整个batch的mask，用于mask掉非原子token
        for tokens, s in zip(tokens_list, tokenized_input):
            atom_keep = [t not in non_atomic_tokens for t in tokens]
            atom_keeps.append(atom_keep)

            mask = torch.zeros(s.shape, dtype=torch.bool)
            inputs.append(pad(s, pad=(0, seq_len - s.shape[0]), value=tokenizer.vocabulary[tokenizer.special_tokens['pad']]).to(device))
            masks.append(pad(mask, pad=(0, seq_len - s.shape[0]), value=True).to(device))

        encoder_input = torch.stack(inputs).T  # [seq_len, batch]
        encoder_pad_mask = torch.stack(masks).T  # [seq_len, batch]
        batch = {
            "encoder_input": encoder_input,
            "encoder_pad_mask": encoder_pad_mask,
        }

        # === 模型前向 ===
        outputs = model.encode(batch)  # [seq_len, batch, hidden]
        # print(outputs.shape)
        outputs = outputs.permute(1, 0, 2)  # [batch, seq_len, hidden]
        pad_mask = ~batch['encoder_pad_mask'].T  # [batch, seq_len]

        for j, (out, mask, keep, pdbid, num_atoms, map_order) in enumerate(zip(outputs, pad_mask, atom_keeps, pdbid_list, num_atoms_list, map_order_list)):
            valid_out = out[mask]  # 去除 padding
            try:
                atom_feature = valid_out[keep]
                # print(map_order)
                atom_feature = atom_feature[map_order]
                if atom_feature.shape[0] != num_atoms:
                    log_file.write(f"{pdbid}: atom_feature.shape={atom_feature.shape} vs num_atoms={num_atoms}\n")
                    continue
                np.save(os.path.join(output_dir, f"{pdbid}.npy"), atom_feature.cpu().numpy())
            except Exception as e:
                log_file.write(f"{pdbid}: Exception during processing - {str(e)}\n")
