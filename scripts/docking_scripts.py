from torch_geometric.data import DataLoader
import torch.distributions as D
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolTransforms
from rdkit import rdBase
import glob
import os

from vina import Vina

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["TBB_NUM_THREADS"] = "1"
import numpy as np
import torch as th
th.set_num_threads(1)

from scipy.optimize import basinhopping, brute, differential_evolution

import copy

from tqdm import tqdm

# set the random seeds for reproducibility
np.random.seed(123)
th.cuda.manual_seed_all(123)
th.manual_seed(123)
import sys
sys.path.append("/data1/***/codes/GenScore-main")
from torch.utils.data import DataLoader
from BioLM_Score.data.data2 import VSDataset
from BioLM_Score.model.utils2 import run_an_eval_epoch, collate_fn
from BioLM_Score.model.model4 import BioLLMScore, GraphTransformer, GatedGCN

import pandas as pd
from BioLM_Score.feats.mol2graph_rdmda_res import label_query
from docking_utils import get_score_model, optimize_conformation, compute_euclidean_distances_matrix, apply_changes, \
    calculate_probablity, get_pdbqt_string_from_mol, get_center_and_size

import pandas as pd
from BioLM_Score.feats.mol2graph_rdmda_res import label_query
from docking_utils import get_score_model
device = 'cuda'
score_model = get_score_model(modelpath='/data1/***/codes/GenScore-main/scripts/mmgatedgcn_ft_1.0_04.pth')
affinity_csv = "/data1/***/codes/GenScore-main/data/train_data/pdbbind_affinity.csv"
affinity_pd = pd.read_csv(affinity_csv, index_col=0, header=0)


def vina_refine_pose_from_pdbqt_string(
    receptor_pdbqt: str,
    ligand_pdbqt_string: str,
    box_center,          # [cx, cy, cz]
    box_size,            # [sx, sy, sz]
    out_pose_pdbqt: str,
    cpu: int = 1,
    seed: int = 42,
):
    """
    对“当前给定的 ligand pose(PDBQT)”做 Vina 局部优化（微调），并写出精修后的 pose。
    返回：(score_before, score_after, energy_components_before, energy_components_after)
    """
    v = Vina(sf_name="vina", cpu=cpu, seed=seed)

    # 1) 载入受体与配体（配体用 string 形式可避免频繁落盘）
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_string(ligand_pdbqt_string)

    # 2) 计算 vina maps（box center / size 必须与你打分时一致）
    v.compute_vina_maps(center=list(box_center), box_size=list(box_size))

    # 3) 精修前打分
    e_before = v.score()          # numpy array: [total, inter, intra, ...]（版本相关）
    score_before = float(e_before[0])

    # 4) 局部优化（微调）
    e_after = v.optimize()        # optimize() 会更新内部 pose
    score_after = float(e_after[0])

    # 5) 写出精修后的单个 pose
    v.write_pose(out_pose_pdbqt, overwrite=True)

    return score_before, score_after

import subprocess
from pathlib import Path

def pdbqt_to_pdb_with_obabel(pdbqt_path: str, sdf_path: str, add_h: bool=False, del_h: bool=False) -> None:
    pdbqt_path = str(Path(pdbqt_path).resolve())
    sdf_path = str(Path(sdf_path).resolve())

    cmd = ["/data1/***/anaconda3/envs/rtmscore/bin/obabel", "-ipdbqt", pdbqt_path, "-osdf", "-O", sdf_path]

    if add_h and del_h:
        raise ValueError("add_h and del_h cannot both be True.")
    if add_h:
        cmd.append("-h")
    if del_h:
        cmd.append("-d")

    env = os.environ.copy()

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"obabel failed:\n{proc.stdout}")



def dock_compound(data, pdbid, ligand_path, protein_pdbqt, out_path, dist_threshold=5.0, popsize=150):
    sample = data[0]
    _, gp, gl, protein_embs, ligand_embs, _ = collate_fn([sample])

    gl, gp = gl.to(device), gp.to(device)
    protein_embs = protein_embs.to(device)
    ligand_embs = ligand_embs.to(device)

    score_model.eval()
    with th.no_grad():
        pi, sigma, mu, dist, *_ = score_model(gl, gp, protein_embs, ligand_embs)
    real_mol = Chem.MolFromMolFile(ligand_path, removeHs=False)
    # real_mol = Chem.AddHs(real_mol)
    opt = optimize_conformation(mol=real_mol, target_coords=gp.pos.cpu(), n_particles=1,
                               pi=pi.cpu(), mu=mu.cpu(), sigma=sigma.cpu(), dist_threshold=dist_threshold)
    #Define bounds
    flatten_pos = gp.pos.reshape(-1, 3)
    mask = ~th.isnan(flatten_pos).any(dim=1)
    protein_atom_pos = flatten_pos[mask]
    max_bound = np.concatenate([[np.pi]*3, protein_atom_pos.cpu().max(0)[0].numpy(), [np.pi]*len(opt.rotable_bonds)], axis=0)
    min_bound = np.concatenate([[-np.pi]*3, protein_atom_pos.cpu().min(0)[0].numpy(), [-np.pi]*len(opt.rotable_bonds)], axis=0)
    bounds = (min_bound, max_bound)

    scores = []  # 保存每一代的最优得分

    # def record_best(xk, convergence):
    #     # xk 是当前最优解的参数，convergence 是一个收敛指标
    #     val = opt.score_conformation(xk)
    #     scores.append(val)

    # Optimize conformations
    print("%s start optimize..."%(pdbid))
    result = differential_evolution(opt.score_conformation, list(zip(bounds[0],bounds[1])), maxiter=800,
                                    popsize=8,
                                    mutation=(0.5, 1), recombination=0.8, disp=False, seed=123, workers=1)
    print("%s end optimize..."%(pdbid))

    # Get optimized molecule and RMSD
    opt_mol = apply_changes(opt.mol, result['x'], opt.rotable_bonds)

    ligand_pdbqt_string = get_pdbqt_string_from_mol(opt_mol)
    center, _ = get_center_and_size(real_mol)
    size  = (max_bound - min_bound)[3:6]
    pdbqt_output_file = os.path.join(out_path,pdbid+'.pdbqt')

    score_before, score_after = vina_refine_pose_from_pdbqt_string(protein_pdbqt, ligand_pdbqt_string, center, size, pdbqt_output_file)
    # sdf_out_file = os.path.join(out_path, pdbid+'.sdf')
    # pdbqt_to_pdb_with_obabel(pdbqt_output_file, sdf_out_file)
    # docked_mol = Chem.MolFromMolFile(sdf_out_file, removeHs=True)

    result['num_MixOfGauss'] = th.where(dist <= dist_threshold)[0].size(0)
    result['before_rmsd'] = Chem.rdMolAlign.AlignMol(opt_mol, real_mol, atomMap=list(zip(opt.noHidx,opt.noHidx)))
    # result['after_rmsd'] = Chem.rdMolAlign.AlignMol(docked_mol, real_mol, atomMap=list(zip(opt.noHidx,opt.noHidx)))
    result['pdb_id'] = pdbid
    # Get score of real conformation
    ligCoords = th.stack([th.tensor(m.GetConformer().GetPositions()[opt.noHidx]) for m in [real_mol]])
    dist = compute_euclidean_distances_matrix(ligCoords, opt.targetCoords.reshape(1,-1,3)).flatten().unsqueeze(1)
    score_real_mol = calculate_probablity(opt.pi, opt.sigma, opt.mu, dist)
    score_real_mol[th.where(dist > dist_threshold)[0]] = 0.
    # if th.any(dist < 1.5):
    #     score_real_mol[th.where(dist < 1.5)[0]] += th.log(dist/1.5)
    result['score_real_mol']  = score_real_mol.reshape(opt.n_particles, -1).sum(1).item()
    result['vina_score_before'] = score_before
    result['vina_score_after'] = score_after
    del ligCoords, dist, score_real_mol

    result['pkx'] = label_query(pdbid, affinity_pd)
    result['num_atoms'] = real_mol.GetNumHeavyAtoms()
    result['num_rotbonds'] = len(opt.rotable_bonds)
    result['rotbonds'] = opt.rotable_bonds
    #result['num_MixOfGauss'] = mu.size(0)

    # opt_mol = Chem.RemoveHs(opt_mol)
    # writer = Chem.SDWriter("/data1/***/codes/GenScore-main/docking_out/docking_out_0106/%s.sdf" % (pdbid))
    # writer.write(opt_mol)
    # writer.close()

    return result



pdbids = [x for x in os.listdir("/data1/***/codes/RTMScore-main/data/CASF-2016/coreset") if os.path.isdir("/data1/***/codes/RTMScore-main/data/CASF-2016/coreset/%s"%(x))]
results = []
for pdbid in tqdm(pdbids):
    if pdbid != '1pxn':
        continue
    pocket_path = "/data1/***/codes/RTMScore-main/data/CASF-2016/coreset/%s/%s_pocket_chain_10A.pdb"%(pdbid, pdbid)
    ligand_path = "/data1/***/codes/RTMScore-main/data/CASF-2016/coreset/%s/%s_ligand_sort.sdf"%(pdbid, pdbid)
    pocket_emb = "/data1/***/codes/esm-main/data/protein_embeddings/casf_embeddings/pocket_embeddings/%s.npy"%(pdbid)
    ligand_emb = "/data1/***/codes/Chemformer-main/data/ligand_embeddings3/casf2016/%s.npy"%(pdbid)
    protein_pdbqt = f"/data1/***/codes/RTMScore-main/data/CASF-2016/coreset/{pdbid}/{pdbid}_protein.pdbqt"
    out_path = "/data1/***/codes/GenScore-main/docking_out/para_analysis/iter800"
    data = VSDataset(ligs=ligand_path,
					prot=pocket_path,
					 protein_emb=pocket_emb,
					 ligand_emb=ligand_emb,
					cutoff=10,
					explicit_H=False,
					use_chirality=True,
					parallel=False)

    try:
        results.append(dock_compound(data, pdbid, ligand_path,protein_pdbqt, out_path))
        d = {}
        for k in results[0].keys():
            if k != 'jac':
                d[k] = tuple(d[k] for d in results)
        # th.save(d, 'DockingResults_CASF2016_CoreSet.chk')
        results_df = pd.DataFrame.from_dict(d)
        results_df.to_csv(out_path + 'DockingResults_CASF2016_CoreSet.csv', index=False)
    except:
        print("%s error!"%pdbid)
