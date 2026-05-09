import argparse

import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
import pickle
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
sys.path.append("/data1/***/codes/GenScore-main")
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from BioLM_Score.data.data2 import PDBbindDataset
from BioLM_Score.model.model4 import BioLLMScore, GraphTransformer, GatedGCN
from BioLM_Score.model.utils2 import run_an_eval_epoch,collate_fn
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
args['seeds'] = 126
args["num_workers"] = 10
args["model_path"] = args1.model_path
args["data_dir"] = "/data1/***/codes/GenScore-main/data/train_data3"
args["test_prefix"] = "v2020_casf"
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


def scoring(ids, prots, ligs, modpath,
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
	data = PDBbindDataset(ids=ids,prots=prots,ligs=ligs,
						  protein_embs='/data1/***/codes/esm-main/data/protein_embeddings/pdbbindv2020_embeddings/pocket_embeddings/',
						  ligand_embs='/data1/***/codes/Chemformer-main/data/ligand_embeddings3/pdbbindv2020/')
						
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
	return data.pdbids, preds
	#except:
	#	print("failed to scoring for {} and {}".format(prot, lig))
	#	return None, None

def obtain_score_metrics(df):
	#Calculate the Pearson correlation coefficient
	regr = linear_model.LinearRegression()
	regr.fit(df.score.values.reshape(-1,1), df.logKa.values.reshape(-1,1))
	preds = regr.predict(df.score.values.reshape(-1,1))
	rp = pearsonr(df.logKa, df.score)[0]
	#rp = df[["logKa","score"]].corr().iloc[0,1]
	mse = mean_squared_error(df.logKa, preds)
	num = df.shape[0]
	sd = np.sqrt((mse*num)/(num-1))
	#return rp, sd, num
	print("The regression equation: logKa = %.2f + %.2f * Score"%(float(regr.coef_), float(regr.intercept_)))
	print("Number of favorable sample (N): %d"%num)
	print("Pearson correlation coefficient (R): %.3f"%rp)
	print("Standard deviation in fitting (SD): %.2f"%sd)


def cal_PI(score, logKa):
	"""Define the Predictive Index function"""
	logKa, score = zip(*sorted(zip(logKa, score), key=lambda x: x[0], reverse=False))
	W = []
	WC = []
	for i in np.arange(0, 5):
		for j in np.arange(i + 1, 5):
			w_ij = abs(logKa[i] - logKa[j])
			W.append(w_ij)
			if score[i] < score[j]:
				WC.append(w_ij)
			elif score[i] > score[j]:
				WC.append(-w_ij)
			else:
				WC.append(0)

	pi = float(sum(WC)) / float(sum(W))
	return pi

def obtain_rank_metrics(df):
	df_groupby = testdf.groupby('target')

	spearman = df_groupby.apply(lambda x: x[["logKa", "score"]].corr("spearman").iloc[1, 0])
	kendall = df_groupby.apply(lambda x: x[["logKa", "score"]].corr("kendall").iloc[1, 0])
	PI = df_groupby.apply(lambda x: cal_PI(x.score, x.logKa))

	print("The Spearman correlation coefficient (SP): %.3f" % spearman.mean())
	print("The Kendall correlation coefficient (tau): %.3f" % kendall.mean())
	print("The Predictive index (PI): %.3f" % PI.mean())


prots='%s/%s_prot.pt'%(args["data_dir"],args["test_prefix"])
ligs='%s/%s_lig.pt'%(args["data_dir"],args["test_prefix"])
ids='%s/%s_ids.npy'%(args["data_dir"],args["test_prefix"])

#data = VSDataset(ids=ids,graphs=graphs)

_, preds = scoring(ids, prots, ligs, args["model_path"], parallel=True, **args)

df = pd.read_csv("/data1/***/codes/RTMScore-main/data/CASF-2016/power_scoring/CoreSet.dat", sep='[,,\t, ]+', header=0, engine='python')
df_score = pd.DataFrame(zip(np.load(ids)[0],preds), columns=["#code","score"])
testdf = pd.merge(df,df_score,on='#code')
testdf[["#code","score"]].to_csv("/data1/***/codes/RTMScore-main/data/CASF-2016/power_scoring/examples/2025-4-29/%s.dat"%args["outprefix"], index=False, sep="\t")

obtain_score_metrics(testdf)
print('*********************************************')
obtain_rank_metrics(testdf)






