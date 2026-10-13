"""Reuse validated evaluators; only protein-LM mapping and CPU bound are adapted."""
import argparse
from collections import Counter,defaultdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid
import numpy as np

RUNTIME=Path(__file__).resolve().parent
sys.path.insert(0,str(RUNTIME))
from protein_lm_shuffle import ProteinLMShuffleBank as OriginalBank,_stable_seed

def main():
    p=argparse.ArgumentParser(add_help=False); p.add_argument('--benchmark',choices=['casf','dekois'],required=True); p.add_argument('--graph-workers',type=int,default=6)
    a,rest=p.parse_known_args(); path=RUNTIME/('casf_joint_family_worker.py' if a.benchmark=='casf' else 'dekois_strict_controls_worker.py')
    spec=importlib.util.spec_from_file_location('validated_worker',path); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    if '--help' in rest or '-h' in rest:
        sys.argv=[str(path),*rest]; module.main(); return
    from joblib import Parallel as RealParallel
    import BioLM_Score.data.data2 as dataset_module
    def bounded(*args,**kwargs):
        requested=kwargs.pop('n_jobs',a.graph_workers); kwargs['n_jobs']=a.graph_workers if requested<0 else min(requested,a.graph_workers)
        return RealParallel(*args,**kwargs)
    dataset_module.Parallel=bounded
    if a.benchmark=='casf' and a.graph_workers>1:
        original_make=module.make_dataset
        def parallel_make(VSDataset,*args,**kwargs):
            def construct(**options):
                options['parallel']=True
                return VSDataset(**options)
            return original_make(construct,*args,**kwargs)
        module.make_dataset=parallel_make
    # CASF has five complexes per target; exclude donors from the same target cluster.
    groups={}
    if a.benchmark=='casf':
        project=Path(rest[rest.index('--project-root')+1]); table=module.read_casf_table(project/'data/CASF-2016/power_scoring/CoreSet.dat','#code')
        groups=dict(zip(table['#code'].astype(str),table['target'].astype(str)))
    class Bank(OriginalBank):
        def __init__(self,embedding_paths,seed=20261009):
            super().__init__(embedding_paths,seed)
            grouped=defaultdict(list)
            for key in self.original: grouped[groups.get(key,key)].append(key)
            order=sorted(grouped); rng=np.random.default_rng(_stable_seed(seed,'cross:group-block-order')); rng.shuffle(order)
            blocks=[]; owner=[]; recipients=[]; widths=[]; slices={}; cursor=0
            for gi,group in enumerate(order):
                keys=sorted(grouped[group]); arrays=[]
                for key in keys:
                    arr=self.original[key]; slices[key]=slice(cursor,cursor+len(arr)); cursor+=len(arr); recipients.append(key); arrays.append(arr)
                block=np.concatenate(arrays); blocks.append(block[rng.permutation(len(block))]); owner.extend([gi]*len(block)); widths.append(len(block))
            pool=np.concatenate(blocks); owners=np.asarray(owner); offset=max(widths)
            if 2*offset>len(pool): raise ValueError('test largest target group too large')
            donor=np.roll(pool,-offset,axis=0); donor_owners=np.roll(owners,-offset)
            if np.any(owners==donor_owners): raise AssertionError('same test target donor')
            self.cross={key:np.ascontiguousarray(donor[slices[key]]) for key in recipients}
            self.cross_owner_counts={key:dict(Counter(str(order[int(i)]) for i in donor_owners[slices[key]])) for key in recipients}
            self.cross_offset=offset
        def audit(self):
            result=super().audit(); result['target_group_definition']='CASF target cluster' if a.benchmark=='casf' else 'DEKOIS target ID'; result['test_mapping']='fixed, no label use'; return result
        def write_audit(self,path):
            path=Path(path); path=path.with_name(path.stem+f'_seed{self.seed}'+path.suffix); path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp'); tmp.write_text(json.dumps(self.audit(),indent=2)+'\n'); tmp.replace(path)
    module.ProteinLMShuffleBank=Bank
    sys.argv=[str(path),*rest]
    # DEKOIS parses its own graph-workers option; retain the same bound.
    if a.benchmark=='dekois': sys.argv+=['--graph-workers',str(a.graph_workers)]
    module.main()

if __name__=='__main__': main()
