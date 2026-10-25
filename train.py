"""Train TGRL with PPO-based Graph-Counterfactual Optimization (GCO)."""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import DRL
from env.case_generator import CaseGenerator
from env.fjsp_env import FJSPEnv
from validate import validate, get_validate_env


def setup_seed(seed):
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # torch.use_deterministic_algorithms(True)


def main(config_path='config.json'):
    with open(config_path,encoding='utf-8') as f: config=json.load(f)
    env_paras=copy.deepcopy(config['env_paras'])
    model_paras=copy.deepcopy(config['model_paras'])
    train_paras=config['train_paras']
    setup_seed(train_paras.get('seed',300))
    requested=train_paras.get('device','auto')
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested)
    if device.type=='cuda': torch.cuda.set_device(device)
    if device.type=='cpu': torch.set_num_threads(train_paras.get('cpu_threads',4))
    env_paras['device']=device; model_paras['device']=device
    model=DRL.GCOTrainer(model_paras,train_paras,env_paras['batch_size'])
    print('Model device:',next(model.policy.parameters()).device,flush=True)
    validation_paras=copy.deepcopy(env_paras)
    validation_paras['batch_size']=env_paras['valid_batch_size']
    env_valid=get_validate_env(validation_paras)
    memory=DRL.Memory()
    j,m,f=env_paras['num_jobs'],env_paras['num_mas'],env_paras['num_fac']
    low,high=max(1,int(0.8*m)),max(1,int(1.2*m))
    timestamp=time.strftime('%Y%m%d_%H%M%S')
    save=Path(train_paras.get('output_dir','save'))/f'train_{j}_{m}_{f}_{timestamp}'
    save.mkdir(parents=True,exist_ok=False)
    (save/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    viz=None
    if train_paras.get('viz',False):
        from visdom import Visdom
        viz=Visdom(env=train_paras.get('viz_name','TGRL'))
    best=float('inf'); best_file=None; records=[]; validation_rows=[]; instance_rows=[]
    env=None; started=time.time()
    maximum=train_paras['max_iterations']
    if min(maximum,train_paras['parallel_iter'],train_paras['update_timestep'],train_paras['save_timestep'])<1:
        raise ValueError('Training intervals must be positive')
    for iteration in range(1,maximum+1):
        if env is None or (iteration-1)%train_paras['parallel_iter']==0:
            nums=[random.randint(low,high) for _ in range(j)]
            case=CaseGenerator(j,m,f,low,high,nums_ope=nums)
            env=FJSPEnv(case,env_paras)
            print(f'Iteration {iteration}: J={j}, M={m}, F={f}, operations={sum(nums)}',flush=True)
        state=env.state; rollout_start=time.time()
        while not bool(env.done_batch.all()):
            with torch.no_grad(): actions=model.policy_old.act(state,memory,env.done_batch)
            state,rewards,dones=env.step(actions)
            memory.rewards.append(rewards.detach().cpu().clone())
            memory.is_terminals.append(dones.detach().cpu().clone())
        makespan=env.makespan_batch.mean().item()
        if train_paras.get('check_feasibility',False) and not env.validate_gantt()[0]:
            raise RuntimeError(f'Infeasible training schedule: {env.validation_errors[:5]}')
        env.reset()
        if iteration%train_paras['update_timestep']==0 or iteration==maximum:
            loss,reward=model.update(memory,env_paras,train_paras); memory.clear_memory()
            record=dict(iteration=iteration,makespan=makespan,reward=reward,loss=loss,
                        seconds=time.time()-rollout_start,**model.last_metrics)
            records.append(record); pd.DataFrame(records).to_csv(save/'training.csv',index=False)
            print(f'Iteration {iteration}: makespan={makespan:.3f}, reward={reward:.3f}, '
                  f'loss={loss:.4f}, beta={model.last_metrics["cf_beta"]:.3f}, '
                  f'KL={model.last_metrics["kl"]:.5f}, time={record["seconds"]:.2f}s',flush=True)
            if viz is not None:
                viz.line(X=np.array([iteration]),Y=np.array([reward]),win='reward',update='append')
        if iteration%train_paras['save_timestep']==0 or iteration==maximum:
            value,values=validate(validation_paras,env_valid,model.policy_old)
            validation_rows.append(dict(iteration=iteration,makespan=value.item()))
            instance_rows.append([iteration,*values.cpu().tolist()])
            if value.item()<best:
                best=value.item()
                new_file=save/f'save_best_{j}_{m}_{f}_{iteration}.pt'
                torch.save(model.policy.state_dict(),new_file)
                if best_file is not None: best_file.unlink()
                best_file=new_file
            print(f'Validation: {value.item():.3f} (best: {best:.3f})',flush=True)
            pd.DataFrame(validation_rows).to_excel(save/'training_ave.xlsx',index=False)
            columns=['iteration']+[f'instance_{i+1}' for i in range(env_valid.batch_size)]
            pd.DataFrame(instance_rows,columns=columns).to_excel(save/'training_instances.xlsx',index=False)
    print(f'Finished in {time.time()-started:.2f}s. Best checkpoint: {best_file}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config.json')
    main(parser.parse_args().config)
