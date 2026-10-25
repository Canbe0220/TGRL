"""Evaluate TGRL by greedy decoding or independent full-trajectory sampling."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import DRL
from env.fjsp_env import FJSPEnv
from env.load_data import nums_detec
from train import setup_seed


def resolve_checkpoints(model_dir, rules):
    """TGRL selects all checkpoints; DRL/CALM remain legacy selector aliases."""
    model_dir=Path(model_dir)
    if not isinstance(rules,list) or not rules or not all(isinstance(r,str) for r in rules):
        raise ValueError('test_paras.rules must be ["TGRL"] or a nonempty list of .pt filenames')
    selectors={'TGRL','DRL','CALM'}
    if any(r in selectors for r in rules):
        if len(rules)!=1:
            raise ValueError('Use one model selector or explicit .pt filenames; do not mix them')
        checkpoints=sorted(model_dir.glob('*.pt'))
    else:
        checkpoints=[model_dir/r for r in rules]
    if not checkpoints:
        raise FileNotFoundError(f'Put trained TGRL .pt state dictionaries in {model_dir}')
    for checkpoint in checkpoints:
        if checkpoint.suffix!='.pt' or not checkpoint.is_file():
            raise FileNotFoundError(f'TGRL checkpoint not found: {checkpoint}')
    return checkpoints


@torch.no_grad()
def schedule(env,model,memories=None,flag_sample=False):
    policy=model.policy_old if hasattr(model,'policy_old') else model
    policy.eval(); state=env.reset()
    device=next(policy.parameters()).device
    if device.type=='cuda': torch.cuda.synchronize(device)
    started=time.perf_counter()
    while not bool(env.done_batch.all()):
        actions=policy.act(state,memories,env.done_batch,flag_sample=flag_sample,flag_train=False)
        state,_,_=env.step(actions)
    if device.type=='cuda': torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    if not env.validate_gantt()[0]: raise RuntimeError(f'Invalid test schedule: {env.validation_errors[:5]}')
    return env.makespan_batch.detach().clone(),elapsed


def main(config_path='config.json'):
    with open(config_path,encoding='utf-8') as f: config=json.load(f)
    ep=copy.deepcopy(config['env_paras']); mp=copy.deepcopy(config['model_paras'])
    tp=config['test_paras']; train=config['train_paras']
    setup_seed(tp.get('seed',300))
    request=tp.get('device','auto')
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if request=='auto' else request)
    if device.type=='cuda': torch.cuda.set_device(device)
    else: torch.set_num_threads(train.get('cpu_threads',4))
    ep['device']=device;mp['device']=device
    folder=Path(tp.get('data_root','data_test'))/tp['data_path']
    files=sorted(folder.glob('*.fjs'))[:tp['num_ins']]
    if not files: raise FileNotFoundError(f'No .fjs files in {folder}')
    model_dir=Path(tp.get('model_dir','model'))
    checkpoints=resolve_checkpoints(model_dir,tp.get('rules',['TGRL']))
    policy=DRL.TGRLPolicy(mp).to(device)
    print('Model device:',next(policy.parameters()).device,flush=True)
    output=Path(tp.get('output_dir','save'))/time.strftime('test_%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    results={'file_name':[p.name for p in files]}; timings=copy.deepcopy(results)
    sample=bool(tp.get('sample',False)); repeats=tp.get('num_average',1)
    ep['batch_size']=tp.get('num_sample',100) if sample else 1
    if ep['batch_size']<1 or repeats<1: raise ValueError('Positive sampling/repetition counts required')
    for checkpoint in checkpoints:
        # These files are our own tensor-only state dictionaries.
        try: weights=torch.load(checkpoint,map_location=device,weights_only=True)
        except TypeError: weights=torch.load(checkpoint,map_location=device)
        policy.load_state_dict(weights,strict=True); policy.eval()
        values=[]; durations=[]
        for index,path in enumerate(files):
            jobs,machines,factories,_,_=nums_detec(path.read_text().splitlines())
            ep.update(num_jobs=jobs,num_mas=machines,num_fac=factories)
            env=FJSPEnv([str(path)]*ep['batch_size'],ep,data_source='file')
            scores=[]; times=[]
            for _ in range(repeats):
                result,seconds=schedule(env,policy,flag_sample=sample)
                scores.append(result.min().item() if sample else result.item());times.append(seconds)
            values.append(float(np.mean(scores)));durations.append(float(np.mean(times)))
            print(f'{checkpoint.name} [{index+1}/{len(files)}] {path.name}: makespan={values[-1]:.3f}',flush=True)
        results[checkpoint.name]=values;timings[checkpoint.name]=durations
        pd.DataFrame(results).to_excel(output/'makespan.xlsx',index=False)
        pd.DataFrame(timings).to_excel(output/'time.xlsx',index=False)
        print(f'{checkpoint.name}: mean makespan={np.mean(values):.3f}, mean time={np.mean(durations):.3f}s',flush=True)
    print(f'Results: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.json')
    main(parser.parse_args().config)
