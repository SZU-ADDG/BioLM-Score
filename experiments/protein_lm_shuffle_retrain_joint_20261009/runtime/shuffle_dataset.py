"""Shuffle only protein-LM rows; graph geometry, masks and ligand remain intact."""
import hashlib
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

def stable_seed(seed, token):
    return int.from_bytes(hashlib.sha256(f'{seed}:{token}'.encode()).digest()[:8],'little')

class ShuffleDataset(Dataset):
    def __init__(self, ids, labels, gp, gl, protein, ligand, metadata, indices, control, seed, split):
        self.ids=ids; self.labels_all=labels; self.gp=gp; self.gl=gl
        self.protein=protein; self.ligand=ligand; self.meta=metadata
        self.indices=np.asarray(indices,dtype=np.int64); self.control=control; self.seed=seed; self.split=split
        self.labels=torch.tensor(labels[self.indices],dtype=torch.float32)
        self.mapping=None; self.set_epoch(0)
    def __len__(self): return len(self.indices)
    def set_epoch(self,epoch):
        rng=np.random.default_rng(stable_seed(self.seed,f'{self.control}:{self.split}:epoch{epoch}'))
        po=self.meta['protein']['offsets']; total=po[-1]
        mapping=np.full(total,-1,dtype=np.int64)
        if self.control=='within_pocket_lm_shuffle':
            fixed=0; singletons=0
            for i in self.indices:
                start,end=po[i:i+2]; n=end-start; identity=np.arange(n)
                if n==1: permutation=identity; singletons+=1
                else:
                    for _ in range(10000):
                        permutation=rng.permutation(n)
                        if not np.any(permutation==identity): break
                    else: raise RuntimeError('derangement failed')
                mapping[start:end]=start+permutation; fixed+=int(np.sum(permutation==identity))
            detail={'fixed_points':fixed,'unshufflable_single_residue_pockets':singletons}
        elif self.control=='cross_target_lm_shuffle':
            grouped=defaultdict(list)
            for i in self.indices: grouped[self.meta['target_groups'][i]].append(int(i))
            groups=sorted(grouped); rng.shuffle(groups)
            recipient=[]; source=[]; owner=[]; widths=[]
            for group_index,group in enumerate(groups):
                slots=np.concatenate([np.arange(po[i],po[i+1],dtype=np.int64) for i in grouped[group]])
                recipient.append(slots); source.append(slots[rng.permutation(len(slots))]); owner.append(np.full(len(slots),group_index,dtype=np.int32)); widths.append(len(slots))
            recipient=np.concatenate(recipient); source=np.concatenate(source); owner=np.concatenate(owner); offset=max(widths)
            if 2*offset>len(source): raise ValueError('largest target group too large for derangement')
            donor=np.roll(source,-offset); donor_owner=np.roll(owner,-offset)
            if np.any(donor_owner==owner): raise AssertionError('same-target donor')
            if len(np.unique(donor))!=len(source): raise AssertionError('donor replacement')
            mapping[recipient]=donor
            detail={'same_target_donor_slots':0,'source_vectors_used_once':True,'target_groups':len(groups),'rotation_offset':offset}
        elif self.control=='full':
            for i in self.indices: mapping[po[i]:po[i+1]]=np.arange(po[i],po[i+1])
            detail={'identity_mapping':True}
        else: raise ValueError(self.control)
        self.mapping=mapping
        self.audit={'control':self.control,'shuffle_seed':self.seed,'epoch':epoch,'split':self.split,'recipient_count':len(self.indices),'mapping_sha256':hashlib.sha256(mapping.tobytes()).hexdigest(),'donor_pool':'same split only; no train/validation mixing',**detail}
    def __getitem__(self,j):
        i=int(self.indices[j]); po=self.meta['protein']['offsets']; lo=self.meta['ligand']['offsets']
        rows=self.mapping[po[i]:po[i+1]]
        if np.any(rows<0): raise AssertionError('missing shuffle rows')
        pe=torch.from_numpy(np.array(self.protein[rows],dtype=np.float32,copy=True))
        le=torch.from_numpy(np.array(self.ligand[lo[i]:lo[i+1]],dtype=np.float32,copy=True))
        return self.ids[i],self.gp[i],self.gl[i],pe,le,self.labels[j]
