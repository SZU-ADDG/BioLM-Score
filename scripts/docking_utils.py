import contextlib
import copy
import re
import subprocess
import tempfile

import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem, rdDistGeom, rdForceFieldHelpers
from torch.distributions import Normal
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate


sys.path.append("/data1/chenby/codes/GenScore-main")
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from BioLM_Score.data.data2 import VSDataset
from BioLM_Score.model.utils2 import run_an_eval_epoch, collate_fn
from BioLM_Score.model.model4 import BioLLMScore, GraphTransformer, GatedGCN

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def write_pdbqt_string_to_tmp(pdbqt_string: str) -> str:
    """把 PDBQT 字符串写入临时文件，返回路径。调用者负责删除。"""
    fd, tmp_path = tempfile.mkstemp(suffix=".pdbqt", prefix="lig_")
    with os.fdopen(fd, "w") as f:   # 用 os.fdopen 接管文件描述符
        f.write(pdbqt_string)
    return tmp_path

def run_vina_score_only_cmd(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center,  # (cx, cy, cz)
    size,    # (sx, sy, sz)
    vina_exe: str = None,
    extra_args: list = None
) -> float:
    """
    用命令行 vina --score_only 打分，返回 kcal/mol。
    同时兼容 Vina 1.2+ 的 "Affinity:" 和 smina 风格的 "REMARK VINA RESULT:" 输出。
    """
    vina_exe = vina_exe or os.getenv("VINA_EXE", "vina")
    cx, cy, cz = map(float, center)
    sx, sy, sz = map(float, size)

    cmd = [
        vina_exe,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
        "--size_x",   f"{sx:.3f}", "--size_y",   f"{sy:.3f}", "--size_z",   f"{sz:.3f}",
        "--score_only",
        "--cpu", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    m = re.search(r"Affinity:\s*([\-0-9\.]+)\s*\(?kcal/mol\)?", out)
    if m:
        return float(m.group(1))

    m = re.search(r"REMARK VINA RESULT:\s*([\-0-9\.]+)", out)
    if m:
        return float(m.group(1))

    raise RuntimeError(f"未能解析 vina 输出：\n{out}")

def get_score_model(modelpath, **kwargs):
    args = {}
    args["batch_size"] = 128
    args["dist_threhold"] = 5
    args['device'] = 'cuda'
    args["num_workers"] = 10
    args["num_node_featsp"] = 41
    args["num_node_featsl"] = 41
    args["num_edge_featsp"] = 5
    args["num_edge_featsl"] = 10
    args["hidden_dim0"] = 128
    args["hidden_dim"] = 128
    args["n_gaussians"] = 10
    args["dropout_rate"] = 0.15
    args["encoder"] = 'gatedgcn'
    kwargs.update(args)
    if kwargs["encoder"] == "gt":
        ligmodel = GraphTransformer(in_channels=kwargs["num_node_featsl"],
                                    edge_features=kwargs["num_edge_featsl"],
                                    num_hidden_channels=kwargs["hidden_dim0"],
                                    activ_fn=th.nn.SiLU(),
                                    transformer_residual=True,
                                    num_attention_heads=4,
                                    norm_to_apply='batch',
                                    dropout_rate=0.15,
                                    num_layers=6
                                    )

        protmodel = GraphTransformer(in_channels=kwargs["num_node_featsp"],
                                     edge_features=kwargs["num_edge_featsp"],
                                     num_hidden_channels=kwargs["hidden_dim0"],
                                     activ_fn=th.nn.SiLU(),
                                     transformer_residual=True,
                                     num_attention_heads=4,
                                     norm_to_apply='batch',
                                     dropout_rate=0.15,
                                     num_layers=6
                                     )
    elif kwargs["encoder"] == "gatedgcn":
        ligmodel = GatedGCN(in_channels=kwargs["num_node_featsl"],
                            edge_features=kwargs["num_edge_featsl"],
                            num_hidden_channels=kwargs["hidden_dim0"],
                            residual=True,
                            dropout_rate=0.15,
                            equivstable_pe=False,
                            num_layers=6
                            )

        protmodel = GatedGCN(in_channels=kwargs["num_node_featsp"],
                             edge_features=kwargs["num_edge_featsp"],
                             num_hidden_channels=kwargs["hidden_dim0"],
                             residual=True,
                             dropout_rate=0.15,
                             equivstable_pe=False,
                             num_layers=6
                             )
    else:
        raise ValueError("encoder should be \"gt\" or \"gatedgcn\"!")

    model = BioLLMScore(ligmodel, protmodel,
                     in_channels=kwargs["hidden_dim0"],
                     hidden_dim=kwargs["hidden_dim"],
                     n_gaussians=kwargs["n_gaussians"],
                     dropout_rate=kwargs["dropout_rate"],
                     dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])

    checkpoint = th.load(modelpath, map_location=th.device(kwargs['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class optimize_conformation():
    def __init__(self, mol, target_coords, n_particles, pi, mu, sigma, save_molecules=False, dist_threshold=5.0, tau=0.1, grid_center = None,
                 seed=None):
        super(optimize_conformation, self).__init__()
        if seed:
            np.random.seed(seed)

        self.opt_mols = []
        self.n_particles = n_particles
        self.rotable_bonds = get_torsions([mol])
        self.save_molecules = save_molecules
        self.dist_threshold = dist_threshold
        self.mol = get_random_conformation(mol, rotable_bonds=self.rotable_bonds, seed=seed)
        # self.mol = mol

        self.targetCoords = torch.stack([target_coords for _ in range(n_particles)]).double()
        self.pi = torch.cat([pi for _ in range(n_particles)], axis=0)
        self.sigma = torch.cat([sigma for _ in range(n_particles)], axis=0)
        self.mu = torch.cat([mu for _ in range(n_particles)], axis=0)
        self.noHidx = [idx for idx in range(self.mol.GetNumAtoms()) if self.mol.GetAtomWithIdx(idx).GetAtomicNum() != 1]
        self.tau = tau

    def score_conformation(self, values):
        """
        Parameters
        ----------
        values : numpy.ndarray
            set of inputs of shape :code:`(n_particles, dimensions)`
        Returns
        -------
        numpy.ndarray
            computed cost of size :code:`(n_particles, )`
        """
        if len(values.shape) < 2: values = np.expand_dims(values, axis=0)
        mols = [copy.copy(self.mol) for _ in range(self.n_particles)]

        # Apply changes to molecules
        # apply rotations
        [SetDihedral(mols[m].GetConformer(), self.rotable_bonds[r], values[m, 6 + r]) for r in
         range(len(self.rotable_bonds)) for m in range(self.n_particles)]

        # apply transformation matrix
        [rdMolTransforms.TransformConformer(mols[m].GetConformer(), GetTransformationMatrix(values[m, :6])) for m in
         range(self.n_particles)]

        # Calcualte distances between ligand conformation and target
        ligCoords_list = [torch.tensor(m.GetConformer().GetPositions()[self.noHidx]) for m in mols]
        ligCoords = torch.stack(ligCoords_list).double()
        dist = compute_euclidean_distances_matrix(ligCoords, self.targetCoords.reshape(1,-1,3)).reshape(-1,1)

        # Calculate probability for each ligand-target pair
        prob = calculate_probablity(self.pi, self.sigma, self.mu, dist)
        # weight = torch.sigmoid((self.dist_threshold - dist) / self.tau)  # tau≈0.5~1
        # prob = prob * weight

        prob[torch.where(dist > self.dist_threshold)[0]] = 0.

        # Reshape and sum probabilities
        # prob = prob.reshape(self.n_particles, -1).sum(1)
        prob = prob.reshape(self.n_particles, -1).sum(1)

        # conf_mask = dist < 1.5
        # if conf_mask.sum() > 0:
        #     prob += th.log(dist[conf_mask]/1.5).sum()


        if self.save_molecules: self.opt_mols.append(mols[torch.argmax(prob)])

        # Delete useless tensors to free memory
        del ligCoords_list, ligCoords, mols
        # print(prob)

        return -prob.detach().numpy()


import os, copy, numpy as np
from rdkit.Chem import rdMolTransforms

class optimize_conformation_vina():
    def __init__(
        self,
        mol,
        target_coords,          # 仍保留（可不使用），为了兼容你现有调用
        n_particles,
        receptor_pdbqt: str,
        box_center,
        box_size,
        vina_exe: str = None,
        save_molecules: bool = False,
        seed=None,
        grid_center=None,
        box_out_penalty: float = 999.0,
    ):
        if seed is not None:
            np.random.seed(seed)

        assert n_particles == 1, "这个实现按 n_particles=1 写的；要多粒子需要再改一层循环。"

        self.n_particles = n_particles
        self.rotable_bonds = get_torsions([mol])
        self.save_molecules = save_molecules

        # 随机初始构象（与你原来一致）
        self.mol = get_random_conformation(
            mol, rotable_bonds=self.rotable_bonds, seed=seed
        )

        self.receptor_pdbqt = receptor_pdbqt
        if torch.is_tensor(box_center):
            box_center = box_center.detach().cpu().numpy()
        if torch.is_tensor(box_size):
            box_size = box_size.detach().cpu().numpy()

        self.box_center = np.array(box_center, dtype=float)
        self.box_size = np.array(box_size, dtype=float)
        self.vina_exe = vina_exe or os.getenv("VINA_EXE", "vina")
        self.box_out_penalty = float(box_out_penalty)

        # 保存最优
        self.best_score = float("inf")   # vina affinity 越小越好
        self.best_mol = None

        # 仍保留（如果你后续要混合其它目标）
        self.targetCoords = torch.stack([target_coords for _ in range(n_particles)]).double()

    def score_conformation(self, values: np.ndarray) -> float:
        """
        SciPy differential_evolution 会传 shape=(D,)；这里返回 float
        目标：vina --score_only 的 affinity（kcal/mol），越小越好
        """
        if values.ndim < 2:
            values = np.expand_dims(values, axis=0)  # (1,D)

        mol = copy.copy(self.mol)

        # 1) 扭转
        for r in range(len(self.rotable_bonds)):
            SetDihedral(mol.GetConformer(), self.rotable_bonds[r], float(values[0, 6 + r]))

        # 2) 刚体变换（与你原来一致）
        rdMolTransforms.TransformConformer(
            mol.GetConformer(),
            GetTransformationMatrix(values[0, :6])
        )

        # 3) 盒外直接罚分（否则大量 999 会导致 DE 很快“收敛”提前停）
        if not is_mol_in_box(mol, self.box_center, self.box_size):
            return self.box_out_penalty

        # 4) RDKit mol -> pdbqt -> vina --score_only
        try:
            pdbqt_str = get_pdbqt_string_from_mol(mol)
            lig_tmp = write_pdbqt_string_to_tmp(pdbqt_str)
            try:
                s = run_vina_score_only_cmd(
                    receptor_pdbqt=self.receptor_pdbqt,
                    ligand_pdbqt=lig_tmp,
                    center=self.box_center,
                    size=self.box_size,
                    vina_exe=self.vina_exe,
                    extra_args=None
                )
            finally:
                try:
                    os.remove(lig_tmp)
                except Exception:
                    pass
        except Exception:
            return self.box_out_penalty

        score = float(s)

        # 5) 可选：记录最优构象
        # if self.save_molecules and score < self.best_score:
        #     self.best_score = score
        #     self.best_mol = copy.copy(mol)

        return score


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def GetTransformationMatrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    transMat = np.array([[np.cos(z) * np.cos(y), (np.cos(z) * np.sin(y) * np.sin(x)) - (np.sin(z) * np.cos(x)),
                          (np.cos(z) * np.sin(y) * np.cos(x)) + (np.sin(z) * np.sin(x)), disp_x],
                         [np.sin(z) * np.cos(y), (np.sin(z) * np.sin(y) * np.sin(x)) + (np.cos(z) * np.cos(x)),
                          (np.sin(z) * np.sin(y) * np.cos(x)) - (np.cos(z) * np.sin(x)), disp_y],
                         [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x), disp_z],
                         [0, 0, 0, 1]], dtype=np.double)
    return transMat

#
def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi)
    prob = logprob.exp().sum(1)

    return prob



def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                            or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


# def get_random_conformation(mol, rotable_bonds=None, grid_center=None, seed=None):
#     if isinstance(mol, Chem.Mol):
#         # Check if ligand it has 3D coordinates, otherwise generate them
#         try:
#             mol.GetConformer()
#         except:
#                 mol=Chem.AddHs(mol)
#                 AllChem.EmbedMolecule(mol)
#                 AllChem.MMFFOptimizeMolecule(mol)
#     else:
#         raise Exception('mol should be an RDKIT molecule')
#     if seed:
#             np.random.seed(seed)
#     if rotable_bonds is None:
#         rotable_bonds = get_torsions([mol])
#     # new_conf = apply_changes(mol, np.random.rand(len(rotable_bonds)+6)*10, rotable_bonds)
#
#     vals = np.empty(6 + len(rotable_bonds))
#     vals[:3] = np.random.uniform(-np.pi, np.pi, size=3)  # 欧拉角 rx,ry,rz（弧度）
#     vals[3:6] = np.random.uniform(-3.0, 3.0, size=3)  # 平移 dx,dy,dz（Å）可按需要调为 ±2~5
#     vals[6:] = np.random.uniform(-np.pi, np.pi, size=len(rotable_bonds))  # 每根可旋转键的二面角（弧度）
#
#     new_conf = apply_changes(Chem.Mol(mol), vals, rotable_bonds)  # 用 RDKit 深拷贝更安全
#     Chem.rdMolTransforms.CanonicalizeConformer(new_conf.GetConformer())  #此代码会将分子中心平移到原点
#     # if grid_center is not None: #将初始化的分子移动至口袋中心
#     #     conf = new_conf.GetConformer()
#     #     coords = np.array([list(conf.GetAtomPosition(i)) for i in range(new_conf.GetNumAtoms())])
#     #     current_center = coords.mean(axis=0)
#     #     translation = [grid_center[i] - current_center[i] for i in range(3)]
#     #     for i in range(new_conf.GetNumAtoms()):
#     #         pos = conf.GetAtomPosition(i)
#     #         conf.SetAtomPosition(i, (pos.x + translation[0],
#     #                                  pos.y + translation[1],
#     #                                  pos.z + translation[2]))
#     return new_conf



def get_random_conformation(mol, rotable_bonds=None, seed=None):
    if isinstance(mol, Chem.Mol):
        # Check if ligand it has 3D coordinates, otherwise generate them
        try:
            mol.GetConformer()
        except:
                mol=Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
    else:
        raise Exception('mol should be an RDKIT molecule')
    if seed:
            np.random.seed(seed)
    if rotable_bonds is None:
        rotable_bonds = get_torsions([mol])
    new_conf = apply_changes(mol, np.random.rand(len(rotable_bonds, )+6)*10, rotable_bonds)
    Chem.rdMolTransforms.CanonicalizeConformer(new_conf.GetConformer())
    return new_conf


def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[6 + r]) for r in range(len(rotable_bonds))]

    # apply transformation matrix
    rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol



def suppress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper
@suppress_stdout
def ligand_rdmol_to_pdbqt_string(
    rdmol: Chem.Mol,
    run_etkdg: bool = False,
    run_uff: bool = False,
    use_meeko: bool = True,
) -> str:
    # construct/refine molecular structure
    if run_etkdg or run_uff:
        rdmol = Chem.Mol(rdmol)
    if run_etkdg:
        assert rdmol.GetNumConformers() == 0
        param = rdDistGeom.srETKDGv3()
        param.randomSeed = 1
        param.numThreads = 1
        rdDistGeom.EmbedMolecule(rdmol, param)
    if run_uff:
        assert rdmol.GetNumConformers() == 1
        rdForceFieldHelpers.UFFOptimizeMolecule(rdmol)

    # AllChem.ComputeGasteigerCharges(rdmol)

    # pdbqt conversion
    if use_meeko:
        """Meeko molecular preparation"""
        preparator = MoleculePreparation()
        setup, *_ = preparator.prepare(rdmol)
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        return pdbqt_string
    # else:
    #     """Simple pdbqt conversion with obabel"""
    #     # TODO: check whether following code do work or not.
    #     pbmol: pybel.Molecule = pybel.readstring("sdf", Chem.MolToMolBlock(rdmol))
    #     return pbmol.write("pdbqt")


def get_center_and_size(mol: Chem.Mol, buffer: float = 10.0):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    center = coords.mean(axis=0)
    size = coords.max(axis=0) - coords.min(axis=0) + buffer
    return center.tolist(), size.tolist()


#获取蛋白质口袋的中心坐标
def get_pocket_center(pocket_coords):
    coords = pocket_coords.reshape(-1, 3)
    coords = coords[~torch.isnan(coords).any(dim=1)] #去除包含 NaN 的坐标
    pocket_center = coords.mean(dim=0)
    return pocket_center.tolist()

# def set_ligand_from_rdmol(v, mol):
#     mol_h = Chem.AddHs(mol, addCoords=True)
#     pdbqt = ligand_rdmol_to_pdbqt_string(mol_h)
#     v.set_ligand_from_string(pdbqt)

def get_pdbqt_string_from_mol(mol):
    mol_h = Chem.AddHs(mol, addCoords=True)
    pdbqt = ligand_rdmol_to_pdbqt_string(mol_h)
    return pdbqt


from rdkit import Chem
def prepare_pdbqt_template_from_text(mol_H: Chem.Mol, pdbqt_text: str):
    """
    把一次性的 PDBQT 文本改造成“可回填坐标的模板”（带 {x}{y}{z} 占位符的行列表）。
    - ATOM/HETATM 行：仅替换 XYZ 三列为 {x:8.3f}{y:8.3f}{z:8.3f}
    - ROOT/BRANCH/ENDBRANCH/ENDROOT/TORSDOF 行：原样保留（含换行）
    - 其余（REMARK/MODEL/ENDMDL…）忽略
    """
    atom_template = []
    atom_line_count = 0

    for raw in pdbqt_text.splitlines():
        line = raw.rstrip("\n")
        rec = (line[:6].strip().upper() if len(line) >= 6 else line.strip().upper())

        if rec in ("ATOM", "HETATM"):
            # 确保长度足够，便于定宽切片覆盖到 Z 列
            if len(line) < 60:
                line = line.ljust(60)

            s = list(line)

            # PDBQT 坐标定宽列（0-based）：X[30:38], Y[38:46], Z[46:54]
            def put(span, key):
                a, b = span
                frag = ("{" + f"{key}:8.3f" + "}").rjust(b - a)
                s[a:b] = list(frag)

            put((30, 38), "x")
            put((38, 46), "y")
            put((46, 54), "z")

            # 保持其余列不变；行尾补换行方便后续 join
            atom_template.append("".join(s))
            atom_line_count += 1

        elif rec in ("ROOT", "BRANCH", "ENDBRANCH", "ENDROOT", "TORSDOF"):
            atom_template.append(line)  # 控制行原样保留
        else:
            atom_template.append(line)
            continue  # 其他行忽略

    # 一致性检查
    n_atoms = mol_H.GetNumAtoms()
    if atom_line_count != n_atoms:
        raise ValueError(
            f"模板原子行数({atom_line_count}) != 分子原子数({n_atoms})，"
            "请确认使用的是同一带氢分子生成的 PDBQT 文本。"
        )

    # 再检查占位符是否完整
    ph = sum(1 for t in atom_template if "{x" in t and "{y}"[:2] in t and "{z" in t)
    if ph != n_atoms:
        raise ValueError(f"占位符行数({ph}) != 分子原子数({n_atoms})，模板未正确替换 XYZ 列。")

    return atom_template

def pdbqt_from_template(mol_H_with_conf: Chem.Mol, atom_template):
    """
    给定“带氢分子 + 当前构象”的 RDKit 分子，把 XYZ 坐标回填到模板，生成**单模型**PDBQT字符串。
    约定：
      - atom_template 中，只有 ATOM/HETATM 行含 {x}{y}{z} 占位符；
      - ROOT/BRANCH/ENDROOT/ENDBRANCH/TORSDOF 行原样输出。
    """
    conf = mol_H_with_conf.GetConformer()
    out_lines = []
    atom_i = 0

    for tpl in atom_template:
        if "{x" in tpl and "{y" in tpl and "{z" in tpl:
            p = conf.GetAtomPosition(atom_i)
            out_lines.append(tpl.format(x=p.x, y=p.y, z=p.z))
            atom_i += 1
        else:
            out_lines.append(tpl)

    if atom_i != mol_H_with_conf.GetNumAtoms():
        raise ValueError("回填的原子数与分子原子数不一致，请检查模板和分子是否对应。")

    return "\n".join(out_lines) + "\n"


def compute_euclidean_distances_matrix(X, Y):
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    N_l = X.shape[1]
    X = X.double()
    Y = Y.double()

    dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y ** 2, axis=-1).unsqueeze(1) + th.sum(X ** 2, axis=-1).unsqueeze(-1)
    return th.nan_to_num((dists ** 0.5).view(N_l, -1, 24), 10000).min(axis=-1)[0]
