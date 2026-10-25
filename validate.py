import time
from pathlib import Path
import torch
import DRL
from env.fjsp_env import FJSPEnv


def get_validate_env(env_paras):
    key=f"{env_paras['num_jobs']}{env_paras['num_mas']:02d}{env_paras['num_fac']:02d}"
    folder=Path(env_paras.get('validation_dir',f'data_dev/{key}'))
    files=sorted(str(p) for p in folder.glob('*.fjs'))
    size=env_paras['batch_size']
    if len(files)<size: raise ValueError(f'{folder}: need {size} validation files, found {len(files)}')
    return FJSPEnv(files[:size],env_paras,data_source='file')


@torch.no_grad()
def validate(env_paras,env,model_policy):
    start=time.time(); was_training=model_policy.training
    model_policy.eval(); state=env.reset(); memory=DRL.Memory()
    while not bool(env.done_batch.all()):
        actions=model_policy.act(state,memory,env.done_batch,flag_sample=False,flag_train=False)
        state,_,_=env.step(actions)
    if not env.validate_gantt()[0]:
        raise RuntimeError(f'Invalid validation schedule: {env.validation_errors[:5]}')
    values=env.makespan_batch.detach().clone()
    env.reset(); model_policy.train(was_training)
    print(f'Validation on {len(values)} instances: {time.time()-start:.2f}s',flush=True)
    return values.mean(),values
