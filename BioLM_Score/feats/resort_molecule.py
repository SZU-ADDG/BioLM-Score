import pandas as pd
from rdkit import Chem
import os
import argparse
from pathlib import Path


def resort_mol(dataset_path, output_csv="resorted_mol_info.csv"):
    records = []  # Store one record per processed ligand.

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("_ligand.mol2"):
                mol_file = os.path.join(root, file)
                out_file = os.path.join(root, file.rsplit('.', 1)[0] + '_sort.sdf')

                # Read the MOL2 file.
                mol = Chem.MolFromMol2File(mol_file)
                if mol is None:
                    print(f"Failed to read MOL2 file: {mol_file}")
                    continue

                # Trigger RDKit's canonical SMILES atom-order property.
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
                except Exception as e:
                    print(f"Failed to process {mol_file}: {e}")
                    continue

                # Reorder atoms to match the canonical SMILES order.
                renum_mol = Chem.RenumberAtoms(mol, m_order)

                # Write the reordered molecule.
                w = Chem.SDWriter(out_file)
                w.write(renum_mol)
                w.close()

                # Record the atom-symbol order.
                mol_atoms = [atom.GetSymbol() for atom in renum_mol.GetAtoms()]

                # Record metadata.
                records.append({
                    "pdbid": file.rsplit('_')[0],
                    "smiles": smiles,
                    "atom_sequence": " ".join(mol_atoms)
                })

    # Save the metadata table.
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonicalize ligand atom order for Chemformer features.")
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="Directory containing ligand MOL2 files")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    resort_mol(dataset_path=str(args.dataset_path), output_csv=str(args.output_csv))
