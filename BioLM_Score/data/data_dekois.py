import pandas as pd
import numpy as np
import torch as th
from dgl import load_graphs
from rdkit import Chem
import os
import tempfile
import shutil
from joblib import Parallel, delayed
from torch_geometric.data import Batch, Data, Dataset  #, InMemoryDataset
from ..feats.mol2graph_rdmda_res import prot_to_graph, mol_to_graph, load_mol
from ..feats.extract_pocket_prody import extract_pocket

class PDBbindDataset(Dataset):
	def __init__(self,
				ids=None,
				ligs=None,
				prots=None,
				labels=None,
				 protein_embs=None,
				 ligand_embs=None
				):
		super(PDBbindDataset, self).__init__()
		self.labels = labels
		self.protein_embs = protein_embs
		self.ligand_embs = ligand_embs
		if isinstance(ids,np.ndarray) or isinstance(ids,list):
			self.pdbids = ids
		else:
			try:
				self.pdbids = np.load(ids)
			except:
				raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
			if self.pdbids.shape[0] == 1:
				pass
			elif self.pdbids.shape[0] == 2:
				self.labels = self.pdbids[-1].astype(float)
				self.pdbids = self.pdbids[0]
			else:
				raise ValueError('the file to store numpy.ndarray should have one/two dimensions')

		if isinstance(ligs,np.ndarray) or isinstance(ligs,tuple) or isinstance(ligs,list):
			if isinstance(ligs[0],Data):
				self.gls = ligs
			else:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
		else:
			try:
				self.gls = th.load(ligs)
			except:
				raise ValueError('the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')

		if isinstance(prots,np.ndarray) or isinstance(prots,th.Tensor) or isinstance(prots,list):
			if isinstance(prots[0],Data):
				self.gps = prots
			else:
				raise ValueError('the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')
		else:
			try:
				self.gps = th.load(prots)
			except:
				raise ValueError('the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')

		if isinstance(protein_embs, str):
			self.protein_embs = []
			for i, pid in enumerate(self.pdbids):
				npy_path = os.path.join(protein_embs, f"{pid}.npy")
				if not os.path.exists(npy_path):
					print(f"Protein embedding not found: {npy_path}")
					continue
				emb = np.load(npy_path)
				# protein_x = self.gps[i].x
				# if emb.shape[0]!=protein_x.shape[0]:
				# 	i=0
				self.protein_embs.append(th.tensor(emb, dtype=th.float))

		if isinstance(ligand_embs, str):
			self.ligand_embs = []
			for i, pid in enumerate(self.pdbids):
				npy_path = os.path.join(ligand_embs, f"{pid}.npy")
				if not os.path.exists(npy_path):
					print(f"Failed! ligand embedding not found: {npy_path}")
					continue
				emb = np.load(npy_path)
				self.ligand_embs.append(th.tensor(emb, dtype=th.float))


		self.gls = Batch.from_data_list(self.gls)
		self.gps = Batch.from_data_list(self.gps)
		assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs == len(self.protein_embs) == len(self.ligand_embs)
		if self.labels is None:
			self.labels = th.zeros(len(self.pdbids))
		else:
			self.labels = th.tensor(self.labels)

	def len(self):
		return len(self.pdbids)

	def get(self, idx):
		pdbid = self.pdbids[idx]
		gp = self.gps[idx]
		gl = self.gls[idx]
		label = self.labels[idx]
		prot_emb = self.protein_embs[idx]
		lig_emb = self.ligand_embs[idx]
		return pdbid, gp, gl, prot_emb, lig_emb, label

	def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
		#random.seed(seed)
		np.random.seed(seed)
		if valnum is None:
			valnum = int(valfrac * len(self.pdbids))
		val_inds = np.random.choice(np.arange(len(self.pdbids)),valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)),val_inds)
		return train_inds, val_inds

#
# class PDBbindDataset(Dataset):
# 	def __init__(self,
# 				 ids=None,
# 				 ligs=None,
# 				 prots=None,
# 				 labels=None,
# 				 ):
# 		super(PDBbindDataset, self).__init__()
# 		self.labels = labels
# 		if isinstance(ids, np.ndarray) or isinstance(ids, list):
# 			self.pdbids = ids
# 		else:
# 			try:
# 				self.pdbids = np.load(ids)
# 			except:
# 				raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
# 			if self.pdbids.shape[0] == 1:
# 				pass
# 			elif self.pdbids.shape[0] == 2:
# 				self.labels = self.pdbids[-1].astype(float)
# 				self.pdbids = self.pdbids[0]
# 			else:
# 				raise ValueError('the file to store numpy.ndarray should have one/two dimensions')
#
# 		if isinstance(ligs, np.ndarray) or isinstance(ligs, tuple) or isinstance(ligs, list):
# 			if isinstance(ligs[0], Data):
# 				self.gls = ligs
# 			else:
# 				raise ValueError(
# 					'the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
# 		else:
# 			try:
# 				self.gls = th.load(ligs)
# 			except:
# 				raise ValueError(
# 					'the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
#
# 		if isinstance(prots, np.ndarray) or isinstance(prots, th.Tensor) or isinstance(prots, list):
# 			if isinstance(prots[0], Data):
# 				self.gps = prots
# 			else:
# 				raise ValueError(
# 					'the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')
# 		else:
# 			try:
# 				self.gps = th.load(prots)
# 			except:
# 				raise ValueError(
# 					'the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')
#
# 		self.gls = Batch.from_data_list(self.gls)
# 		self.gps = Batch.from_data_list(self.gps)
# 		assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs
# 		if self.labels is None:
# 			self.labels = th.zeros(len(self.pdbids))
# 		else:
# 			self.labels = th.tensor(self.labels)
#
# 	def len(self):
# 		return len(self.pdbids)
#
# 	def get(self, idx):
# 		pdbid = self.pdbids[idx]
# 		gp = self.gps[idx]
# 		gl = self.gls[idx]
# 		label = self.labels[idx]
# 		return pdbid, gp, gl, label

	def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
		# random.seed(seed)
		np.random.seed(seed)
		if valnum is None:
			valnum = int(valfrac * len(self.pdbids))
		val_inds = np.random.choice(np.arange(len(self.pdbids)), valnum, replace=False)
		train_inds = np.setdiff1d(np.arange(len(self.pdbids)), val_inds)
		return train_inds, val_inds


class VSDataset(Dataset):
	def __init__(self,  
				ids=None,
				ligs=None,
				prot=None,
				 protein_emb=None,
				 ligand_emb=None,
				labels=None,
				gen_pocket=False,
				cutoff=None,
				reflig=None,
				explicit_H=False, 
				use_chirality=True,
				parallel=True			
				):
		super(VSDataset, self).__init__()
		self.labels = labels
		self.gp=None
		self.gls=None
		self.pocketdir = None
		self.prot = None
		self.ligs = None
		self.cutoff = cutoff
		self.explicit_H=explicit_H
		self.use_chirality=use_chirality
		self.parallel=parallel

		self.protein_emb = None
		self.ligand_emb = None
		if isinstance(protein_emb, str):
			self.protein_emb = np.load(protein_emb)
			self.protein_emb = th.tensor(self.protein_emb, dtype=th.float)

		if isinstance(ligand_emb,str):
			try:
				self.ligand_emb = np.load(ligand_emb)
			except:
				self.ligand_emb = np.load(ligand_emb,allow_pickle=True)
			self.ligand_emb = [th.tensor(arr, dtype=th.float32) for arr in self.ligand_emb]
		
		if isinstance(prot, Chem.rdchem.Mol):
			assert gen_pocket == False
			self.prot = prot
			self.gp = prot_to_graph(self.prot, cutoff)
		else:
			if gen_pocket:
				if cutoff is None or reflig is None:
					raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
				try:
					self.pocketdir = tempfile.mkdtemp()
					extract_pocket(prot, reflig, cutoff, 
								protname="temp",
								workdir=self.pocketdir)
					pocket = load_mol("%s/temp_pocket_%s.pdb"%(self.pocketdir, cutoff), 
								explicit_H=explicit_H, use_chirality=use_chirality)
					self.prot = pocket
					self.gp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			else:
				try:
					pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
					#self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)	
					self.prot = pocket
					self.gp = prot_to_graph(self.prot, cutoff)
				except:
					raise ValueError('The graph of pocket cannot be generated')
			
		if isinstance(ligs,np.ndarray) or isinstance(ligs,list):
			if isinstance(ligs[0], Chem.rdchem.Mol):
				self.ligs = ligs
				self.gls = self._mol_to_graph()
			elif isinstance(ligs[0], Data):
				self.gls = ligs
			else:
				raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
		else:
			if ligs.endswith(".mol2"):
				lig_blocks = self._mol2_split(ligs)	
				self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			elif ligs.endswith(".sdf"):
				lig_blocks = self._sdf_split(ligs)	
				self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
				self.gls = self._mol_to_graph()
			else:
				try:	
					self.gls,_ = load_graphs(ligs)
				except:
					raise ValueError('Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')
		
		if ids is None:
			if self.ligs is not None:
				self.idsx = ["%s-%s"%(self.get_ligname(lig),i) for i, lig in enumerate(self.ligs)]
			else:
				self.idsx = ["lig%s"%i for i in range(len(self.gls))]
		else:
			self.idsx = ids
		
		self.ids, self.gls = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.gls)))
		self.ids = list(self.ids)
		self.gls = Batch.from_data_list(self.gls)
		assert len(self.ids) == self.gls.num_graphs == len(self.ligand_emb)
		if self.labels is None:
			self.labels = th.zeros(len(self.ids))
		else:
			self.labels = th.tensor(self.labels)
		
		if self.pocketdir is not None:
			shutil.rmtree(self.pocketdir)

	def len(self):
		return len(self.ids)
	
	def get(self, idx):
		id = self.ids[idx]
		gp = self.gp
		gl = self.gls[idx]
		label = self.labels[idx]
		protein_emb = self.protein_emb
		ligand_emb = self.ligand_emb[idx]
		if gl.x.shape[0] != ligand_emb.shape[0]:
			print(id)
		return id, gp, gl, protein_emb, ligand_emb, label
		
	def _mol2_split(self, infile):
		contents = open(infile, 'r').read()
		return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]
	
	def _sdf_split(self, infile):
		contents = open(infile, 'r').read()
		return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]
	
	def _mol_to_graph0(self, lig):
		try:
			smiles = Chem.MolToSmiles(lig, canonical=True)
			m_order = lig.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
			lig_reordered = Chem.RenumberAtoms(lig, m_order)
			gx = mol_to_graph(lig_reordered, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
		except:
			# print("failed to scoring for {} and {}".format(self.gp, lig))
			return None
		return gx

	def _mol_to_graph(self):
		if self.parallel:
			return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
		else:
			graphs = []
			for lig in self.ligs:
				graphs.append(self._mol_to_graph0(lig))
			return graphs
	
	def get_ligname(self, m):
		if m is None:
			return None
		else:
			if m.HasProp("_Name"):
				return m.GetProp("_Name")
			else:
				return None
	
