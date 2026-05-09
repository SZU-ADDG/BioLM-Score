import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,6,7,8,9"
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
import pickle
sys.path.append("/data1/***/codes/GenScore-main")
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from BioLM_Score.data.data2 import VSDataset
from BioLM_Score.model.utils2 import run_an_eval_epoch, collate_fn
from BioLM_Score.model.model4 import BioLLMScore, GraphTransformer, GatedGCN
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
	parser = argparse.ArgumentParser(description='Model training parameters')

	parser.add_argument('--encoder', type=str, default='gt')
	parser.add_argument('--model_path', type=str, default="/data1/***/codes/RTMScore-main/scripts/xxx.pth",
						help='Path to the model')
	parser.add_argument('--outprefix', type=str, default="rtmscore2", help='Output prefix')

	return parser.parse_args()

args1 = parse_args()
args={}
args["batch_size"] = 128
args["dist_threhold"] = 5.
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
# args['device'] = 'cpu'
args['seeds'] = 126
args["num_workers"] = 10
args["model_path"] = args1.model_path
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
						
	model = BioLLMScore(ligmodel, protmodel,
					in_channels=kwargs["hidden_dim0"], 
					hidden_dim=kwargs["hidden_dim"], 
					n_gaussians=kwargs["n_gaussians"], 
					dropout_rate=kwargs["dropout_rate"], 
					dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])
	
	checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
	model.load_state_dict(checkpoint['model_state_dict']) 
	preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
	return data.ids, preds
	#except:
	#	print("failed to scoring for {} and {}".format(prot, lig))
	#	return None, None



def score_compound(pdbid, ligid, **args):
	return scoring(prot="/data1/***/dataset/pdbbind/PDBbind_v2020_processed/%s/%s_pocket_chain_10A.pdb"%(pdbid, pdbid),
					lig="/data1/***/codes/RTMScore-main/data/CASF-2016/decoys_screening/%s/%s_%s.sdf"%(pdbid, pdbid, ligid),
				   protein_emb='/data1/***/codes/esm-main/data/protein_embeddings/pdbbindv2020_embeddings/pocket_embeddings/%s.npy' % pdbid,
				   ligand_emb="/data1/***/codes/Chemformer-main/data/ligand_embeddings3/casf2016/%s.npy" % ligid,
					modpath=args["model_path"],
					cut=args["cutoff"],
					explicit_H=False, 
					use_chirality=True,
					parallel=True,
					**args
					)


def score_compound2(pdbid, ligids, **args):
	print("%s started....."%pdbid)
	ids_list = []
	scores_list = []
	for ligid in ligids:
		ids, scores = score_compound(pdbid, ligid, **args)
		if ids is not None and scores is not None:
			ids_list.extend(ids)
			scores_list.extend(scores)
	print("%s finished....."%pdbid)
	return pdbid, [ids_list, scores_list]

def score_wrapper(pdbid, ligids, args):
	print("%s started....." % pdbid)
	ids_list = []
	scores_list = []
	for ligid in ligids:
		ids, scores = score_compound(pdbid, ligid, **args)
		if ids is not None and scores is not None:
			ids_list.extend(ids)
			scores_list.extend(scores)
	print("%s finished....." % pdbid)
	return pdbid, [ids_list, scores_list]



ligids = [x for x in os.listdir("/data1/***/codes/RTMScore-main/data/CASF-2016/coreset") if os.path.isdir("/data1/***/codes/RTMScore-main/data/CASF-2016/coreset/%s"%(x))]
pdbids = [x for x in os.listdir("/data1/***/codes/RTMScore-main/data/CASF-2016/decoys_screening") if os.path.isdir("/data1/***/codes/RTMScore-main/data/CASF-2016/decoys_screening/%s"%(x))]
pdbids1 = [x for x in os.listdir("/data1/***/dataset/pdbbind/PDBBind/PDBBind_processed/") if os.path.isdir("/data1/***/dataset/pdbbind/PDBBind/PDBBind_processed/%s"%(x))]
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

outdir = "/data1/***/codes/RTMScore-main/data/CASF-2016/power_screening/examples/2025-4-29/%s"%args["outprefix"]
os.system("mkdir -p %s"%outdir)
for res in results:
	#pdbid = res[0][0].split("_")[0]
	pdbid = res[0]
	df = pd.DataFrame(zip(*res[1]),columns=["#code_ligand_num","score"])
	df["#code_ligand_num"] = df["#code_ligand_num"].str.split("-").apply(lambda x : x[0])
	# print(df)
	df.to_csv("%s/%s_score.dat"%(outdir, pdbid), index=False, sep="\t")

with open("%s_screening.pkl"%args["outprefix"],"wb") as dbFile:
	pickle.dump(results,dbFile)


