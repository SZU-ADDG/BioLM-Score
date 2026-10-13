import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import os
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
import pickle
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from BioLM_Score.data.data2 import VSDataset
from BioLM_Score.model.utils2 import run_an_eval_epoch, collate_fn
from BioLM_Score.model.model4 import BioLLMScore, GraphTransformer, GatedGCN
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
	parser = argparse.ArgumentParser(description='Model training parameters')

	parser.add_argument('--encoder', type=str, choices=['gt', 'gatedgcn'], default='gatedgcn')
	parser.add_argument('--model_type', type=str, choices=['biolm', 'genscore'], default='biolm')
	parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
	parser.add_argument('--outprefix', type=str, default="rtmscore2", help='Output prefix')
	parser.add_argument('--casf-root', type=Path, required=True,
						help='CASF-2016 directory containing coreset/ and decoys_screening/')
	parser.add_argument('--pdbbind-processed-root', type=Path, required=True,
						help='Processed PDBbind complex directory, with one subdirectory per PDB ID')
	parser.add_argument('--protein-embeddings-dir', type=Path, required=True)
	parser.add_argument('--ligand-embeddings-dir', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument('--num-workers', type=int, default=10)

	return parser.parse_args()

args1 = parse_args()
args={}
args["batch_size"] = 128
args["dist_threhold"] = 5.
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
# args['device'] = 'cpu'
args['seeds'] = 126
args["num_workers"] = args1.num_workers
args["model_path"] = args1.model_path
args["model_type"] = args1.model_type
args["cutoff"] = 10.0
args["encoder"] = args1.encoder
args["num_node_featsp"] = 41
args["num_node_featsl"] = 41
args["num_edge_featsp"] = 5
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128 
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.15
args["outprefix"] = args1.outprefix
args["casf_root"] = args1.casf_root
args["pdbbind_processed_root"] = args1.pdbbind_processed_root
args["protein_embeddings_dir"] = args1.protein_embeddings_dir
args["ligand_embeddings_dir"] = args1.ligand_embeddings_dir
args["output_dir"] = args1.output_dir
args["num_workers"] = args1.num_workers

def scoring(prot, lig, protein_emb,ligand_emb, modpath,
			cut=10.0,
			explicit_H=False, 
			use_chirality=True,
			parallel=False,
			**kwargs
			):
	"""
	prot: The input protein file ('.pdb')
	lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
	modpath: The path to store the pre-trained model
	gen_pocket: whether to generate the pocket from the protein file.
	reflig: The reference ligand to determine the pocket.
	cutoff: The distance within the reference ligand to determine the pocket.
	explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
	use_chirality: whether to adopt the information of chirality to represent the molecules.	
	parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
	kwargs: other arguments related with model
	"""
	#try:
	data = VSDataset(ligs=lig,
					prot=prot,
					 protein_emb=protein_emb,
					 ligand_emb=ligand_emb,
					cutoff=cut,		
					explicit_H=explicit_H, 
					use_chirality=use_chirality,
					parallel=parallel)
						
	test_loader = DataLoader(dataset=data, 
							batch_size=kwargs["batch_size"],
							shuffle=False, 
							num_workers=kwargs["num_workers"],
							 collate_fn=collate_fn)
	
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
						
	is_genscore = kwargs["model_type"] == "genscore"
	model = BioLLMScore(ligmodel, protmodel,
					in_channels=kwargs["hidden_dim0"], 
					hidden_dim=kwargs["hidden_dim"], 
					n_gaussians=kwargs["n_gaussians"], 
					dropout_rate=kwargs["dropout_rate"], 
					dist_threhold=kwargs["dist_threhold"],
					use_protein_lm=not is_genscore,
					use_ligand_lm=not is_genscore,
					use_gnn=True).to(kwargs['device'])
	
	checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
	model.load_state_dict(checkpoint['model_state_dict']) 
	preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
	return data.ids, preds



def score_compound(pdbid, ligid, **args):
	casf_root = Path(args["casf_root"])
	pdbbind_root = Path(args["pdbbind_processed_root"])
	return scoring(prot=str(pdbbind_root / pdbid / f"{pdbid}_pocket_chain_10A.pdb"),
					lig=str(casf_root / "decoys_screening" / pdbid / f"{pdbid}_{ligid}.sdf"),
				   protein_emb=str(Path(args["protein_embeddings_dir"]) / f"{pdbid}.npy"),
				   ligand_emb=str(Path(args["ligand_embeddings_dir"]) / f"{ligid}.npy"),
					modpath=args["model_path"],
					cut=args["cutoff"],
					explicit_H=False, 
					use_chirality=True,
					parallel=True,
					**args
					)


def score_compound2(pdbid, ligids, **args):
	ids_list = []
	scores_list = []
	for ligid in ligids:
		ids, scores = score_compound(pdbid, ligid, **args)
		if ids is not None and scores is not None:
			ids_list.extend(ids)
			scores_list.extend(scores)
	return pdbid, [ids_list, scores_list]

def score_wrapper(pdbid, ligids, args):
	ids_list = []
	scores_list = []
	for ligid in ligids:
		ids, scores = score_compound(pdbid, ligid, **args)
		if ids is not None and scores is not None:
			ids_list.extend(ids)
			scores_list.extend(scores)
	return pdbid, [ids_list, scores_list]



casf_root = Path(args["casf_root"])
pdbbind_root = Path(args["pdbbind_processed_root"])
ligids = [p.name for p in (casf_root / "coreset").iterdir() if p.is_dir()]
pdbids = [p.name for p in (casf_root / "decoys_screening").iterdir() if p.is_dir()]
pdbids1 = [p.name for p in pdbbind_root.iterdir() if p.is_dir()]
ids = [pdbid for pdbid in pdbids if pdbid in pdbids1]
task_args = [(pdbid, ligids, args) for pdbid in ids]
if args['device'] == 'cpu':
	with Pool(processes=cpu_count()) as pool:
		results = pool.starmap(score_wrapper, task_args)
else:
	results = []
	for pdbid in ids:
		results.append(score_compound2(pdbid, ligids,**args))
results[:] = [result for result in results if result[0] is not None]

outdir = Path(args["output_dir"]) / args["outprefix"]
outdir.mkdir(parents=True, exist_ok=True)
for res in results:
	#pdbid = res[0][0].split("_")[0]
	pdbid = res[0]
	df = pd.DataFrame(zip(*res[1]),columns=["#code_ligand_num","score"])
	df["#code_ligand_num"] = df["#code_ligand_num"].str.split("-").apply(lambda x : x[0])
	df.to_csv(outdir / f"{pdbid}_score.dat", index=False, sep="\t")

with open(Path(args["output_dir"]) / f"{args['outprefix']}_screening.pkl","wb") as dbFile:
	pickle.dump(results,dbFile)


