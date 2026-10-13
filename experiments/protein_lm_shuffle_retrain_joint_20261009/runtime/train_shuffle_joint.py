"""Joint GatedGCN training with protein-LM-only shuffles, seed-matched splits."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from torch_scatter import scatter_add

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from BioLM_Score.model.model4 import BioLLMScore,GatedGCN
from shuffle_dataset import ShuffleDataset

def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); th.manual_seed(seed); th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark=False; th.backends.cudnn.deterministic=True

def collate(samples):
    pid,gp,gl,pe,le,y=zip(*samples)
    return pid,Batch.from_data_list(gp),Batch.from_data_list(gl),pad_sequence(pe,batch_first=True),pad_sequence(le,batch_first=True),th.stack(y)

def corr(x,y):
    # The ordinary joint Pearson objective, not the target-aware sampler.
    if not bool(th.isfinite(x).all()) or not bool(th.isfinite(y).all()): raise FloatingPointError('non-finite score/label')
    if bool(y.var(unbiased=False)==0): return x.sum()*0.0
    if bool(x.var(unbiased=False)==0): return x.sum()*0.0
    return th.corrcoef(th.stack([x,y]))[1,0]

def forward(model,data):
    ids,gp,gl,pe,le,labels=data; gp=gp.cuda(); gl=gl.cuda(); pe=pe.cuda(); le=le.cuda()
    # GatedGCN updates Batch.x/edge_attr in place. Capture the original
    # chemical labels before the encoder, exactly as the reference utility.
    atom_labels=gl.x[:,:17].argmax(1); bond_labels=gl.edge_attr[:,:4].argmax(1)
    pi,sigma,mu,dist,at,bt,pb=model(gl,gp,pe,le)
    probability=(th.log(pi+1e-10)+Normal(mu,sigma).log_prob(dist.expand_as(mu))).exp().sum(1)
    nll=-th.log(probability+1e-10)
    mdn=nll[dist.squeeze(1)<=7.0].mean()
    keep=dist.squeeze(1)<=5.0
    score=scatter_add(probability[keep],pb[keep].cuda(),dim=0,dim_size=len(ids))
    label=labels.cuda().type_as(score)
    atom=F.cross_entropy(at,atom_labels); bond=F.cross_entropy(bt,bond_labels)
    return score,label,mdn,atom,bond,dist.detach(),pb.detach()

def release(model):
    # These diagnostic attributes are not consumed by the training objective.
    # Removing completed-batch references avoids keeping the previous graph alive.
    for field in ('C','C_mask'):
        if hasattr(model,field): setattr(model,field,None)
    th.cuda.empty_cache()

def evaluate(model,loader):
    model.eval(); scores=[]; labels=[]; sums=defaultdict(float); count=0
    with th.no_grad():
        for data in loader:
            score,label,mdn,atom,bond,dist,pb=forward(model,data); n=len(label); count+=n
            scores.append(score.detach()); labels.append(label.detach())
            for k,v in [('mdn',mdn),('atom',atom),('bond',bond)]: sums[k]+=float(v)*n
            del data,score,label,mdn,atom,bond,dist,pb; release(model)
    c=float(corr(th.cat(scores),th.cat(labels))); result={k:v/count for k,v in sums.items()}
    result.update(corr=c,total=result['mdn']-c+0.001*(result['atom']+result['bond']),count=count)
    if not all(math.isfinite(v) for v in result.values()): raise FloatingPointError('non-finite validation')
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--control',choices=['within_pocket_lm_shuffle','cross_target_lm_shuffle'],required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-data-dir',type=Path,required=True,help='Directory containing v2020_train_{ids,prot,lig} files')
    p.add_argument('--embedding-bank-dir',type=Path,required=True,help='Directory created by prepare_shuffle_assets.py')
    p.add_argument('--shuffle-seed',type=int,default=20261009); p.add_argument('--num-epochs',type=int,default=2000)
    p.add_argument('--preflight',action='store_true'); p.add_argument('--preflight-batches',type=int,default=3)
    p.add_argument('--allocator-fraction',type=float,default=0.48)
    a=p.parse_args(); a.output=a.output.resolve(); a.train_data_dir=a.train_data_dir.resolve(); a.embedding_bank_dir=a.embedding_bank_dir.resolve()
    seed_all(a.seed); th.set_num_threads(1); th.multiprocessing.set_sharing_strategy('file_system')
    if not th.cuda.is_available(): raise RuntimeError('CUDA required')
    th.cuda.set_per_process_memory_fraction(a.allocator_fraction)
    exp=a.output.parent.parent; (exp/'audit').mkdir(exist_ok=True,parents=True)
    meta=json.loads((a.embedding_bank_dir/'metadata.json').read_text())
    ids=meta['ids']; labels=np.array(meta['labels'],dtype=float)
    val=np.sort(np.random.RandomState(a.seed).choice(np.arange(len(ids)),1500,replace=False)); train=np.setdiff1d(np.arange(len(ids)),val)
    if len(train)!=17515 or len(set(ids)&set(meta['casf_excluded'])): raise AssertionError('standard split mismatch')
    split={'train':[ids[i] for i in train],'val':[ids[i] for i in val],'excluded':meta['casf_excluded']}
    fingerprint=hashlib.sha256(json.dumps(split,separators=(',',':'),sort_keys=True).encode()).hexdigest()
    splitdir=exp/'splits'/f'seed{a.seed}'; splitdir.mkdir(parents=True,exist_ok=True)
    for name,values in split.items(): (splitdir/(name+'_ids.txt')).write_text('\n'.join(values)+'\n')
    gp=th.load(a.train_data_dir/'v2020_train_prot.pt',map_location='cpu')
    gl=th.load(a.train_data_dir/'v2020_train_lig.pt',map_location='cpu')
    protein=np.load(a.embedding_bank_dir/'protein.npy',mmap_mode='r'); ligand=np.load(a.embedding_bank_dir/'ligand.npy',mmap_mode='r')
    trainset=ShuffleDataset(ids,labels,gp,gl,protein,ligand,meta,train,a.control,a.shuffle_seed+a.seed,'train')
    valset=ShuffleDataset(ids,labels,gp,gl,protein,ligand,meta,val,a.control,a.shuffle_seed+a.seed,'val')
    prefix=a.output.stem
    atomic(exp/'audit'/(prefix+'_mapping_initial.json'),{'train':trainset.audit,'validation':valset.audit,'train_refresh':'every epoch','validation_refresh':'fixed epoch0','target_definition':meta['target_definition']})
    common=dict(num_hidden_channels=128,dropout_rate=0.15,num_layers=6,residual=True,equivstable_pe=False)
    lm=GatedGCN(in_channels=41,edge_features=10,**common); pm=GatedGCN(in_channels=41,edge_features=5,**common)
    model=BioLLMScore(lm,pm,in_channels=128,hidden_dim=128,n_gaussians=10,dropout_rate=0.15,dist_threhold=7.0,use_protein_lm=True,use_ligand_lm=True,use_gnn=True).cuda()
    optimizer=th.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.00001)
    generator=th.Generator().manual_seed(a.seed)
    loader=DataLoader(trainset,batch_size=64,shuffle=True,generator=generator,num_workers=0,collate_fn=collate)
    vloader=DataLoader(valset,batch_size=64,shuffle=False,num_workers=0,collate_fn=collate)
    config={'model_type':'biolm','stage':'joint','encoder':'gatedgcn','alpha':1.0,'aux_weight':0.001,'batch_size':64,'lr':0.001,'weight_decay':0.00001,'dropout':0.15,'patience':70,'max_epochs':2000,'train_cutoff':7.0,'score_cutoff':5.0,'train_count':17515,'val_count':1500,'seed':a.seed,'control':a.control,'shuffle_seed':a.shuffle_seed+a.seed,'split_fingerprint':fingerprint,'sampler':'ordinary random mini-batches','training_mapping':'deterministic refresh each epoch','validation_mapping':'fixed','initialization':'from scratch; no base or fine-tuning','zero_variance_pearson':'zero affinity term only for zero score or zero label variance'}
    print(json.dumps({'event':'start','config':config,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}),flush=True)
    if a.preflight:
        # Check unmodified graph tensors and ligand embeddings against source-bank rows.
        j=0; i=int(train[j]); item=trainset[j]; origp=protein[meta['protein']['offsets'][i]:meta['protein']['offsets'][i+1]]
        if not th.equal(item[1].x,gp[i].x) or not th.allclose(item[1].pos,gp[i].pos,rtol=0,atol=0,equal_nan=True): raise AssertionError('protein graph changed')
        if not th.allclose(item[2].pos,gl[i].pos,rtol=0,atol=0,equal_nan=True): raise AssertionError('ligand geometry changed')
        batches=list(loader.batch_sampler)
        selected=batches[:a.preflight_batches]
        # Test the largest padded-pair allocation in a real shuffled epoch.
        selected.append(max(batches,key=lambda batch:len(batch)*max(meta['protein']['counts'][train[j]] for j in batch)*max(meta['ligand']['counts'][train[j]] for j in batch)))
        probe=collate([trainset[j] for j in selected[0][:2]])
        # Independent batches avoid reusing graph objects mutated by GatedGCN.
        normal=list(collate([trainset[j] for j in selected[0][:2]])); normal[3]=pad_sequence([th.from_numpy(np.array(protein[meta['protein']['offsets'][train[j]]:meta['protein']['offsets'][train[j]+1]],copy=True)) for j in selected[0][:2]],batch_first=True)
        model.eval()
        with th.no_grad():
            aligned_result=forward(model,normal); perturbed_result=forward(model,probe)
            if not th.equal(aligned_result[5],perturbed_result[5]) or not th.equal(aligned_result[6],perturbed_result[6]): raise AssertionError('distance or pair-index changed')
            if not th.equal(aligned_result[5]<=5.0,perturbed_result[5]<=5.0): raise AssertionError('5A pair mask changed')
        del probe,normal,aligned_result,perturbed_result; release(model)
        th.cuda.reset_peak_memory_stats(); steps=[]
        for batchids in selected:
            batch=collate([trainset[j] for j in batchids]); optimizer.zero_grad(set_to_none=True); model.train()
            score,label,mdn,atom,bond,dist,pb=forward(model,batch); loss=mdn-corr(score,label)+0.001*(atom+bond)
            if not bool(th.isfinite(loss)): raise FloatingPointError('preflight nonfinite')
            loss.backward(); optimizer.step(); steps.append(float(loss))
            del batch,score,label,mdn,atom,bond,dist,pb,loss; release(model)
        result={'passed':True,'config':config,'steps':len(steps),'losses':steps,'peak_allocated_mib':th.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':th.cuda.max_memory_reserved()/2**20,'preserved_geometry_and_ligand':True,'validation_mapping':valset.audit,'training_mapping':trainset.audit,'pid':os.getpid()}
        atomic(a.output,result); print(json.dumps(result),flush=True); return
    start=time.time(); elapsed_before_resume=0.0; best=math.inf; counter=0; best_epoch=0
    writer=SummaryWriter(str(exp/'tensorboard'/prefix))
    resume=exp/'resume'/(prefix+'.pt'); resume.parent.mkdir(exist_ok=True)
    first=1
    if resume.exists():
        state=th.load(resume,map_location='cpu')
        if state['config']!=config: raise ValueError('resume scientific configuration mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for k,v in values.items():
                if th.is_tensor(v): values[k]=v.cuda()
        generator.set_state(state['loader_rng']); th.set_rng_state(state['torch_rng']); th.cuda.set_rng_state(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        first=state['epoch']+1; best=state['best']; counter=state['counter']; best_epoch=state['best_epoch']
        elapsed_before_resume=state.get('elapsed_seconds',0.0)
        print('resume from epoch',first,flush=True)
    last_epoch=first-1
    for epoch in (range(first,a.num_epochs+1) if counter<70 else ()):
        last_epoch=epoch
        trainset.set_epoch(epoch); model.train(); sums=defaultdict(float); seen=0; epochstart=time.time()
        for batchnum,data in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            score,label,mdn,atom,bond,dist,pb=forward(model,data); affinity=-corr(score,label); loss=mdn+affinity+0.001*(atom+bond)
            if not bool(th.isfinite(loss)): raise FloatingPointError(f'nonfinite epoch{epoch} batch{batchnum}')
            loss.backward(); optimizer.step(); n=len(label); seen+=n
            for k,v in [('total',loss),('mdn',mdn),('affinity',affinity),('atom',atom),('bond',bond)]: sums[k]+=float(v.detach())*n
            del data,score,label,mdn,atom,bond,dist,pb,affinity,loss; release(model)
            if batchnum==1 or batchnum%100==0:
                atomic(exp/'worker_status'/(prefix+'.json'),{'stage':'training','pid':os.getpid(),'epoch':epoch,'batch':batchnum,'counter':counter,'best_epoch':best_epoch,'updated':time.time(),'config':config})
        validation=evaluate(model,vloader); improved=validation['total']<best
        if improved:
            best=validation['total']; best_epoch=epoch; counter=0
            tmp=a.output.with_suffix('.pth.tmp'); th.save({'model_state_dict':model.state_dict(),'timestep':epoch,'training_metadata':config,'validation':validation},tmp); tmp.replace(a.output)
        else: counter+=1
        train_metrics={k:v/seen for k,v in sums.items()}
        record={'event':'epoch','epoch':epoch,'train':train_metrics,'validation':validation,'best_epoch':best_epoch,'counter':counter,'seconds':time.time()-epochstart,'peak_allocated_mib':th.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':th.cuda.max_memory_reserved()/2**20,'shuffle_mapping':trainset.audit}
        print(json.dumps(record,sort_keys=True),flush=True)
        for k,v in validation.items(): writer.add_scalar('validation/'+k,v,epoch)
        tmp=resume.with_suffix('.pt.tmp'); th.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'epoch':epoch,'best':best,'counter':counter,'best_epoch':best_epoch,'elapsed_seconds':elapsed_before_resume+time.time()-start,'loader_rng':generator.get_state(),'torch_rng':th.get_rng_state(),'cuda_rng':th.cuda.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},tmp); tmp.replace(resume)
        atomic(exp/'worker_status'/(prefix+'.json'),{'stage':'epoch_complete','pid':os.getpid(),'epoch':epoch,'counter':counter,'best_epoch':best_epoch,'validation':validation,'updated':time.time(),'config':config})
        if counter>=70: break
    writer.close(); state=th.load(a.output,map_location='cpu'); model.load_state_dict(state['model_state_dict']); final=evaluate(model,vloader)
    atomic(a.output.with_suffix('.json'),{'complete':True,'config':config,'best_epoch':best_epoch,'epochs_ran':last_epoch,'elapsed_seconds':elapsed_before_resume+time.time()-start,'checkpoint_sha256':sha(a.output),'final_validation':final,'validation_mapping':valset.audit})
    atomic(exp/'worker_status'/(prefix+'.json'),{'stage':'complete','updated':time.time(),'pid':os.getpid(),'config':config})
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__': main()
