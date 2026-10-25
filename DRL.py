"""TGRL: Timetable Graph Reinforcement Learning.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from embedding import TRGEncoder
from transport_graph import build_trg


class Memory:
    """Detached rollout snapshots, including full behavior probabilities and Q."""
    def __init__(self):
        self.states=[]; self.batch_idxes=[]; self.logprobs=[]; self.rewards=[]
        self.is_terminals=[]; self.action_indexes=[]; self.values=[]
        self.old_probs=[]; self.q_values=[]

    def clear_memory(self):
        for value in vars(self).values():
            if isinstance(value,list): value.clear()


def mlp(input_dim,hidden,output,layers=3):
    if layers < 1: raise ValueError('MLP layers must be positive')
    modules=[]
    for k in range(layers-1):
        modules.extend((nn.Linear(input_dim if k==0 else hidden,hidden),nn.Tanh()))
    modules.append(nn.Linear(hidden if layers>1 else input_dim,output))
    return nn.Sequential(*modules)


class TGRLPolicy(nn.Module):
    """TRG encoder with joint operation-machine actor, state value and action Q.
    """
    def __init__(self,model_paras):
        super().__init__()
        d=model_paras.get('hidden_dim',32)
        self.use_timetable=model_paras.get('use_timetable_graph',True)
        self.store_on_cpu=model_paras.get('store_rollout_on_cpu',True)
        self.encoder=TRGEncoder(d,model_paras.get('attention_heads',4),
                                            model_paras.get('graph_layers',2))
        self.context=nn.Sequential(nn.Linear(3*d,d),nn.Tanh())
        self.actor=mlp(2*d,model_paras.get('n_latent_actor',64),1,model_paras.get('n_hidden_actor',3))
        self.critic=mlp(3*d,model_paras.get('n_latent_critic',64),1,model_paras.get('n_hidden_critic',3))
        self.action_critic=mlp(2*d,model_paras.get('n_latent_critic',64),1,model_paras.get('n_hidden_critic',3))
        # Initially use the physically interpretable relaxed return prior.
        nn.init.zeros_(self.action_critic[-1].weight); nn.init.zeros_(self.action_critic[-1].bias)
        nn.init.orthogonal_(self.actor[-1].weight,gain=0.01); nn.init.zeros_(self.actor[-1].bias)

    def forward(self,graph):
        actions,pooled=self.encoder(graph)
        context=self.context(pooled)[:,None,None].expand_as(actions)
        readout=torch.cat((actions,context),-1)
        logits=self.actor(readout).squeeze(-1).transpose(1,2).flatten(1)
        mask=graph['eligible'].transpose(1,2).flatten(1)
        logits=logits.masked_fill(~mask,float('-inf'))
        value=self.critic(pooled).squeeze(-1)
        q=(self.action_critic(readout).squeeze(-1)+graph['q_prior']).transpose(1,2).flatten(1)
        return logits,value,q

    def get_action_prob(self,state,memories,flag_sample=False,flag_train=False):
        graph=build_trg(state,self.use_timetable)
        logits,value,q=self(graph)
        probs=torch.softmax(logits,dim=-1)
        if flag_train:
            storage='cpu' if self.store_on_cpu else value.device
            memories.states.append({k:v.detach().to(storage).clone() for k,v in graph.items()})
            memories.batch_idxes.append(state.batch_idxes.detach().cpu().clone())
            memories.values.append(value.detach().cpu().clone())
            memories.q_values.append(q.detach().cpu().clone())
            memories.old_probs.append(probs.detach().cpu().clone())
        return probs,torch.minimum(state.ope_step_batch,state.end_ope_biases_batch),value

    def act(self,state,memories,dones,flag_sample=True,flag_train=True):
        probs,steps,_=self.get_action_prob(state,memories,flag_sample,flag_train)
        distribution=Categorical(probs=probs)
        action=distribution.sample() if flag_sample else probs.argmax(-1)
        jobs=action % steps.size(1)
        machines=action // steps.size(1)
        operations=steps[state.batch_idxes,jobs]
        if flag_train:
            memories.logprobs.append(distribution.log_prob(action).detach().cpu().clone())
            memories.action_indexes.append(action.detach().cpu().clone())
        return torch.stack((operations,machines,jobs))

    def evaluate(self,graph,actions):
        logits,value,q=self(graph)
        dist=Categorical(logits=logits)
        return dist.log_prob(actions),value,dist.entropy(),dist.probs,q


# Fields with an operation axis; J and M may vary between inference instances,
# but a PPO update must contain one J/M configuration.
_OPERATION_FIELDS={'tgrl_op','op_mask','job_id','prev','next','prev_mask','next_op_mask','om_mask','proc_norm'}


def collate_graphs(states):
    max_n=max(s['tgrl_op'].size(1) for s in states)
    output={}
    shapes={(s['tgrl_pair'].size(1),s['tgrl_pair'].size(2)) for s in states}
    if len(shapes)!=1: raise ValueError('Each PPO update must use one J/M configuration')
    for key in states[0]:
        values=[]
        for state in states:
            x=state[key].cpu()
            if key in _OPERATION_FIELDS and x.size(1)<max_n:
                shape=list(x.shape); shape[1]=max_n-x.size(1)
                x=torch.cat((x,x.new_zeros(shape)),dim=1)
            values.append(x)
        output[key]=torch.cat(values,dim=0)
    return output


def compute_gae(memory,gamma,lam):
    """Ragged active batches and multiple complete episodes; no dummy transitions."""
    if not memory.states or len(memory.rewards)!=len(memory.states):
        raise ValueError('Incomplete rollout records')
    if not bool(memory.is_terminals[-1].all()):
        raise ValueError('Collect complete episodes before update (truncated rollout needs bootstrap)')
    envs=memory.rewards[0].numel()
    next_value=torch.zeros(envs); next_gae=torch.zeros(envs)
    advantages=[None]*len(memory.states); targets=[None]*len(memory.states)
    for t in range(len(memory.states)-1,-1,-1):
        idx=memory.batch_idxes[t]
        values=memory.values[t].float().reshape(-1)
        rewards=memory.rewards[t].detach().cpu().float()[idx]
        alive=(~memory.is_terminals[t].detach().cpu().bool()[idx]).float()
        delta=rewards+gamma*next_value[idx]*alive-values
        gae=delta+gamma*lam*next_gae[idx]*alive
        advantages[t]=gae; targets[t]=gae+values
        next_value=torch.zeros(envs).index_copy(0,idx,values)
        next_gae=torch.zeros(envs).index_copy(0,idx,gae)
    return torch.cat(advantages),torch.cat(targets)


def gco_surrogate(probs,actions,old_logp,advantages,control,beta,eps_clip):
    """GCO: PPO on the sampled residual plus the exact all-action correction.

    control is frozen and centered under the behavior policy. Without clipping:
      E_old[r*(A-beta*c)] + beta*sum_a pi_new(a)*c(a) = E_old[r*A].
    """
    logp=probs.gather(1,actions[:,None]).squeeze(1).clamp_min(1e-30).log()
    ratio=(logp-old_logp).exp()
    residual=advantages-beta*control.gather(1,actions[:,None]).squeeze(1)
    clipped=torch.minimum(ratio*residual,ratio.clamp(1-eps_clip,1+eps_clip)*residual)
    correction=beta*(probs*control).sum(-1)
    return clipped+correction


class GCOTrainer:
    """PPO-based Graph-Counterfactual Optimization for TGRL.

    """
    def __init__(self,model_paras,train_paras,num_envs=None):
        self.device=torch.device(model_paras.get('device','cpu'))
        self.policy=TGRLPolicy(model_paras).to(self.device)
        self.policy_old=copy.deepcopy(self.policy).eval()
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=train_paras['lr'],
                                         betas=tuple(train_paras.get('betas',(0.9,0.999))))
        self.gamma=train_paras.get('gamma',1.0)
        self.lam=train_paras.get('gae_lambda',0.98)
        self.K_epochs=train_paras.get('K_epochs',3)
        self.minibatch_size=train_paras.get('minibatch_size',512)
        self.eps_clip=train_paras.get('eps_clip',0.2)
        self.A_coeff=train_paras.get('A_coeff',1.0)
        self.vf_coeff=train_paras.get('vf_coeff',0.5)
        self.entropy_coeff=train_paras.get('entropy_coeff',0.01)
        self.q_coeff=train_paras.get('q_coeff',0.25)
        self.cf_coeff=train_paras.get('cf_coeff',0.3)
        self.cf_warmup=train_paras.get('cf_warmup_updates',20)
        self.target_kl=train_paras.get('target_kl',0.02)
        self.max_grad_norm=train_paras.get('max_grad_norm',1.0)
        self.num_envs=num_envs
        self.updates=0; self.last_metrics={}
        # Use statistics from PREVIOUS rollouts. Fitting the normalization on
        # these same sampled actions would weaken the control-variate identity.
        self.adv_mean=0.0; self.adv_scale=1.0
        if self.minibatch_size<1 or self.K_epochs<1: raise ValueError('Positive minibatch size and PPO epochs required')

    def update(self,memory,env_paras=None,train_paras=None):
        advantages,returns=compute_gae(memory,self.gamma,self.lam)
        data=collate_graphs(memory.states)
        actions=torch.cat(memory.action_indexes).long()
        old_logp=torch.cat(memory.logprobs).float()
        old_probs=torch.cat(memory.old_probs).float()
        old_q=torch.cat(memory.q_values).float()
        std=max(self.adv_scale,1e-3)
        norm_adv=(advantages-self.adv_mean)/std
        # The control variate, its scale and beta remain fixed throughout update.
        control=(old_q-(old_probs*old_q).sum(-1,keepdim=True))/std
        control=control.clamp(-5.0,5.0)
        control=control-(old_probs*control).sum(-1,keepdim=True)
        beta=self.cf_coeff*min(1.0,(self.updates+1)/max(self.cf_warmup,1))
        total=actions.numel(); accumulated=0.0; seen=0; stopped=False
        metrics={key:0.0 for key in ('policy_loss','value_loss','q_loss','entropy','kl')}
        self.policy.train()
        for epoch in range(self.K_epochs):
            for indices in torch.randperm(total).split(self.minibatch_size):
                graph={k:v[indices].to(self.device) for k,v in data.items()}
                a=actions[indices].to(self.device)
                adv=norm_adv[indices].to(self.device)
                target=returns[indices].to(self.device)
                oldp=old_probs[indices].to(self.device)
                oldlp=old_logp[indices].to(self.device)
                cv=control[indices].to(self.device)
                logp,values,entropy,probs,q=self.policy.evaluate(graph,a)
                kl=(oldp*(oldp.clamp_min(1e-30).log()-probs.clamp_min(1e-30).log())).sum(-1).mean()
                if self.target_kl>0 and seen and kl.item()>1.5*self.target_kl:
                    stopped=True; break
                policy_loss=-gco_surrogate(probs,a,oldlp,adv,cv,beta,self.eps_clip).mean()
                value_loss=F.mse_loss(values,target)
                q_loss=F.smooth_l1_loss(q.gather(1,a[:,None]).squeeze(1),target)
                # Normalize entropy only by the number of feasible actions.
                feasible=graph['eligible'].flatten(1).sum(-1).float()
                entropy=(entropy/feasible.clamp_min(2).log()).mean()
                loss=(self.A_coeff*policy_loss+self.vf_coeff*value_loss+
                      self.q_coeff*q_loss-self.entropy_coeff*entropy)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite PPO gradient')
                self.optimizer.step()
                count=len(indices); seen+=count; accumulated+=loss.item()*count
                for key,value in zip(metrics,(policy_loss,value_loss,q_loss,entropy,kl)):
                    metrics[key]+=value.item()*count
            if stopped: break
        self.updates+=1
        self.policy_old.load_state_dict(self.policy.state_dict()); self.policy_old.eval()
        residual=norm_adv-beta*control.gather(1,actions[:,None]).squeeze(1)
        self.last_metrics={k:v/max(seen,1) for k,v in metrics.items()}
        self.last_metrics.update(cf_beta=beta,transitions=total,early_stop=stopped,
                                 residual_variance=residual.var(unbiased=False).item(),
                                 advantage_variance=norm_adv.var(unbiased=False).item())
        self.adv_mean=0.9*self.adv_mean+0.1*advantages.mean().item()
        self.adv_scale=0.9*self.adv_scale+0.1*advantages.std(unbiased=False).item()
        # Report complete episode returns rather than padded or duplicated terms.
        sums=torch.zeros(memory.rewards[0].numel()); episode_returns=[]
        for idx,reward,done in zip(memory.batch_idxes,memory.rewards,memory.is_terminals):
            sums[idx]+=reward.detach().cpu()[idx]
            terminal=idx[done.detach().cpu()[idx].bool()]
            episode_returns.extend(sums[terminal].tolist()); sums[terminal]=0
        mean_return=sum(episode_returns)/max(len(episode_returns),1)
        return accumulated/max(seen,1),mean_return
