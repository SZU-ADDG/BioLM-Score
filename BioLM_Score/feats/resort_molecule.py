import pandas as pd
from rdkit import Chem
import os
import argparse
from pathlib import Path


def resort_mol(dataset_path, output_csv="resorted_mol_info.csv"):
    records = []  # 用于保存所有结果

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("_ligand.mol2"):
                mol_file = os.path.join(root, file)
                out_file = os.path.join(root, file.rsplit('.', 1)[0] + '_sort.sdf')

                # 读取 mol2
                mol = Chem.MolFromMol2File(mol_file)
                if mol is None:
                    print(f"读取失败: {mol_file}")
                    continue

                # 触发 smiles 顺序属性
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
                except Exception as e:
                    print(f"处理失败: {mol_file}, 原因: {e}")
                    continue

                # 重排原子顺序
                renum_mol = Chem.RenumberAtoms(mol, m_order)
                # Chem.SanitizeMol(renum_mol)

                # 写入 SDF 文件
                w = Chem.SDWriter(out_file)
                w.write(renum_mol)
                w.close()

                # 提取原子符号顺序
                mol_atoms = [atom.GetSymbol() for atom in renum_mol.GetAtoms()]

                # 记录信息
                records.append({
                    "pdbid": file.rsplit('_')[0],
                    "smiles": smiles,
                    "atom_sequence": " ".join(mol_atoms)
                })

    # 保存为 CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"已保存信息至: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonicalize ligand atom order for Chemformer features.")
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="Directory containing ligand MOL2 files")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    resort_mol(dataset_path=str(args.dataset_path), output_csv=str(args.output_csv))
