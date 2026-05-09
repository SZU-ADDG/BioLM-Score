import os

from rdkit import Chem
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



def get_ligand_atom_features(sdf_path, model, tokenizer, device, seq_len, non_atomic_tokens):
    from rdkit import Chem

    def sdf_split(infile):
        contents = open(infile, 'r').read()
        return [c + "$$$$\n" for c in contents.split("$$$$\n") if c.strip()]

    lig_blocks = sdf_split(sdf_path)
    smiles_list = []

    for i, lig_block in enumerate(lig_blocks):
        mol = Chem.MolFromMolBlock(lig_block)
        if mol is None:
            print(f"Warning: Molecule {i} in {sdf_path} could not be parsed.")
            smiles_list.append(None)
            continue
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
            mol = Chem.RenumberAtoms(mol, m_order)
            smiles_list.append(smiles)
        except Exception as e:
            print(f"Error processing ligand {i}: {e}")
            smiles_list.append(None)

    # 过滤 None
    valid_indices = [i for i, s in enumerate(smiles_list) if s is not None]
    smiles_list = [smiles_list[i] for i in valid_indices]
    lig_features = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]

        tokenized_input = tokenizer.encode(batch_smiles)
        tokens_list = tokenizer.tokenize(batch_smiles)

        inputs, masks, atom_keeps = [], [], []

        for tokens, s in zip(tokens_list, tokenized_input):
            atom_keep_indices = [t not in non_atomic_tokens for t in tokens]
            atom_keeps.append(atom_keep_indices)

            mask = torch.zeros(s.shape, dtype=torch.bool)
            inputs.append(pad(s, pad=(0, seq_len - s.shape[0]),
                              value=tokenizer.vocabulary[tokenizer.special_tokens['pad']]).to(device))
            masks.append(pad(mask, pad=(0, seq_len - s.shape[0]), value=True).to(device))

        encoder_input = torch.stack(inputs).T
        encoder_pad_mask = torch.stack(masks).T
        batch = {
            "encoder_input": encoder_input,
            "encoder_pad_mask": encoder_pad_mask,
        }
        with torch.no_grad():
            outputs = model.encode(batch).permute(1, 0, 2)
        pad_mask = ~batch['encoder_pad_mask'].T

        for out, mask, keep in zip(outputs, pad_mask, atom_keeps):
            valid_out = out[mask]
            assert len(valid_out) == len(keep), f"{sdf_path} Mismatch in token vs keep: {len(valid_out)} vs {len(keep)}"
            atom_feature = valid_out[keep]
            lig_features.append(atom_feature.cpu().numpy())
    assert len(lig_features) == len(smiles_list)

    return lig_features


root_dir = "/data1/***/codes/GenScore-main/data/posebusters_benchmark_set/"
out_path = "/data1/***/codes/GenScore-main/data/posebusters_embs/ligand_embs"


for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        ligand_path = os.path.join(subdir_path, f"{subdir}_ligand.sdf")
        lig_features = get_ligand_atom_features(ligand_path, model, tokenizer, device, seq_len, non_atomic_tokens)

        # save_path = os.path.join(out_path, f"{subdir}_ligand_features.npy")
        # np.save(save_path, np.array(lig_features, dtype=object))

        feat = np.asarray(lig_features[0], dtype=np.float32)  # shape: (num_atoms, hidden_dim)

        save_path = os.path.join(out_path, f"{subdir}_ligand_features.npy")
        np.save(save_path, feat)  # 纯数值 ndarray，不是 object