
import os
import argparse
import sys
from pathlib import Path

import MDAnalysis as mda

import pandas as pd
import numpy as np
from rdkit import Chem

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from BioLM_Score.feats.extract_pocket_prody import extract_pocket




METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA',
        "CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
        "TM","LU","HF","ZR","CE","U","PU","TH"]

aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
    # ESM3中保留的四种非标准氨基酸
    'SEC':'U','ORN':'O'
}

# def pdb_to_sequence(prot):
#     s = []
#     u = mda.Universe(prot)
#     # prev_chain = None
#     for res in u.residues:
#         if res.resname == 'HOH' or res.resname == 'NA':
#             continue
#         # curr_chain = res.segid.strip()
#         # if prev_chain is not None and curr_chain != prev_chain:
#         #     s.append('|')  # 插入链分隔符
#         resn = aa_codes.get(res.resname, 'X')
#         s.append(resn)
#         # prev_chain = curr_chain
#
#     return ''.join(s)

def pdb_to_sequence(prot):
    s= []
    mol = Chem.MolFromPDBFile(prot, removeHs=True, sanitize=False)
    u = mda.Universe(mol)
    for res in u.residues:
        if res.resname == 'HOH' or res.resname == 'NA':
            continue
        resn = aa_codes.get(res.resname, 'X')
        s.append(resn)

    return ''.join(s)


def get_embeddings_from_sequence(sequence, pdbid, client, failed_pdbids_file):
    try:
        # 创建ESMProtein对象
        protein = ESMProtein(sequence=sequence)

        # 将蛋白质序列转换为tensor
        protein_tensor = client.encode(protein)

        # 获取embeddings
        logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        embeddings = logits_output.embeddings
        embeddings = embeddings.squeeze(0)

        assert embeddings.shape[0] == len(sequence) + 2, \
            f"Error: embeddings size mismatch for {pdbid}. Expected {len(sequence) + 2}, got {embeddings.shape[0]}"

        embeddings = embeddings[1:-1]

        # 构建过滤 mask：True 表示该位置不是“|”
        keep_mask = [aa != '|' for aa in sequence]

        assert len(keep_mask) == embeddings.shape[0], \
            f"Length mismatch: {len(keep_mask)} vs embeddings {embeddings.shape[0]}"

        # 过滤掉 | 所在位置
        filtered_embeddings = embeddings[keep_mask]

        # 保存embeddings为npy文件
        return filtered_embeddings
    except Exception as e:
        print(f"Error processing {pdbid}: {e}")
        # 将失败的 pdbid 写入失败文件
        with open(failed_pdbids_file, 'a') as file:
            file.write(f"{pdbid}:{e}\n")


def get_kept_indices(full_pdb_path, pocket_pdb_path):

    full_mol = Chem.MolFromPDBFile(full_pdb_path, removeHs=True, sanitize=False)
    u_full = mda.Universe(full_mol)
    pocket_mol = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=True, sanitize=False)
    u_pocket = mda.Universe(pocket_mol)

    # 获取唯一标识符（链ID, resid, resname, icode）
    def get_residue_ids(universe):
        return [
            (res.segid.strip(), res.resid, res.resname.strip(), getattr(res, 'icode', '').strip())  # 某些版本字段不同)
            # (res.segid.strip(), res.resid, res.resname.strip())
            for res in universe.residues if (res.resname != 'HOH' and res.resname != 'NA')
        ]


    full_residues = get_residue_ids(u_full)
    pocket_residues = get_residue_ids(u_pocket)
    # print(len(pocket_residues))

    # 获取保留和mask残基的序列索引
    kept_indices = []
    masked_indices = []


    pocket_residue_remaining = list(pocket_residues)

    for idx, residue in enumerate(full_residues):
        if residue in pocket_residue_remaining:
            kept_indices.append(idx)
            pocket_residue_remaining.remove(residue)
        else:
            masked_indices.append(idx)

    # print(len(kept_indices))

    assert len(kept_indices) == len(
        pocket_residues), f'{full_pdb_path}:口袋序列特征长度出错！len(kept_indices):{len(kept_indices)},len(pocket_residues):{len(pocket_residues)}'
    return kept_indices

def main():
    parser = argparse.ArgumentParser(description="Generate ESM-C pocket embeddings for prepared protein-ligand folders.")
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Directory containing one prepared complex per subdirectory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="esmc_600m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--failed-log", type=Path, default=Path("get_embedding_failed.txt"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.failed_log.parent.mkdir(parents=True, exist_ok=True)
    client = ESMC.from_pretrained(args.model_name, device=args.device).to(args.device)

    for subdir in os.listdir(args.input_root):
        subdir_path = args.input_root / subdir
        if not subdir_path.is_dir():
            continue
        target_pdb = subdir_path / f"{subdir}_protein.pdb"
        if not target_pdb.is_file():
            print(f"Missing protein file: {target_pdb}")
            continue
        seq = pdb_to_sequence(str(target_pdb))
        protein_embs = get_embeddings_from_sequence(seq, subdir, client, str(args.failed_log))
        if protein_embs is None:
            continue
        pocket_pdb_path = subdir_path / f"{subdir}_pocket_chain_10A.pdb"
        kept_indices = get_kept_indices(str(target_pdb), str(pocket_pdb_path))
        pocket_embs = protein_embs[kept_indices]
        np.save(args.output_dir / f"{subdir}_pocket.npy", pocket_embs.cpu().numpy())


if __name__ == "__main__":
    main()


