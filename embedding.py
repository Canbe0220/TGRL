"""TRG neural encoder: precedence, eligibility, catchability and event factors."""
import math
import torch
from torch import nn
from transport_graph import TGRL_OP_DIM, TGRL_MA_DIM, TGRL_PAIR_DIM, TGRL_NEXT_DIM


def masked_softmax(logits, mask, dim=-1):
    mask = mask.bool()
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=dim)
    weights = weights * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def masked_mean(x, mask, dim):
    return (x*mask[...,None]).sum(dim) / mask.sum(dim).clamp_min(1)[...,None]


def group_context(tokens, groups, valid=None):
    """Linear-memory incidence aggregation; group IDs carry no numerical meaning."""
    b, count, d = tokens.shape
    valid = groups >= 0 if valid is None else valid & (groups >= 0)
    # Graph builder maps departure classes to at most count groups.
    # Factory labels index aggregation slots directly. Extra slots support
    # nonconsecutive labels/empty factories; only equal labels share messages.
    safe = groups.clamp_min(0)
    slots = max(count, int(safe.max().item())+1)
    sums = tokens.new_zeros((b, slots, d))
    numbers = tokens.new_zeros((b, slots, 1))
    sums.scatter_add_(1, safe[...,None].expand(-1,-1,d), tokens*valid[...,None])
    numbers.scatter_add_(1, safe[...,None], valid[...,None].to(tokens.dtype))
    gathered_n = numbers.gather(1, safe[...,None])
    result = (sums/numbers.clamp_min(1)).gather(1, safe[...,None].expand(-1,-1,d))
    return result*valid[...,None], torch.log1p(gathered_n)*valid[...,None]


class RelationLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.d, self.heads, self.width = d, heads, d//heads
        self.oq, self.mk = nn.Linear(d,d), nn.Linear(d,d)
        self.ov, self.mv = nn.Linear(d,d), nn.Linear(d,d)
        self.edge_bias = nn.Linear(1,heads)
        self.pre, self.sub, self.factory = nn.Linear(d,d), nn.Linear(d,d), nn.Linear(d+1,d)
        self.op_update = nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d))
        self.ma_update = nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,d))
        self.onorm, self.mnorm = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, o, m, graph):
        b, n, d = o.shape
        size = m.size(1)
        q = self.oq(o).view(b,n,self.heads,self.width).transpose(1,2)
        k = self.mk(m).view(b,size,self.heads,self.width).transpose(1,2)
        score = q@k.transpose(-1,-2)/math.sqrt(self.width)
        score = score+self.edge_bias(graph['proc_norm'][...,None]).permute(0,3,1,2)
        mask = graph['om_mask'][:,None]
        om = masked_softmax(score,mask,-1)
        mo = masked_softmax(score.transpose(-1,-2),mask.transpose(-1,-2),-1)
        vo = self.ov(o).view(b,n,self.heads,self.width).transpose(1,2)
        vm = self.mv(m).view(b,size,self.heads,self.width).transpose(1,2)
        from_m = (om@vm).transpose(1,2).reshape(b,n,d)
        from_o = (mo@vo).transpose(1,2).reshape(b,size,d)
        predecessor = o.gather(1,graph['prev'][...,None].expand(-1,-1,d))*graph['prev_mask'][...,None]
        successor = o.gather(1,graph['next'][...,None].expand(-1,-1,d))*graph['next_op_mask'][...,None]
        fac, num = group_context(m, graph['factory'])
        o = self.onorm(o+self.op_update(torch.cat((o,from_m,self.pre(predecessor),self.sub(successor)),-1)))
        m = self.mnorm(m+self.ma_update(torch.cat((m,from_o,self.factory(torch.cat((fac,num),-1))),-1)))
        return o*graph['op_mask'][...,None], m


class EventFactorLayer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d+1,d) for _ in range(3)])
        self.gate = nn.Linear(4*d,3)
        self.update = nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d))
        self.norm = nn.LayerNorm(d)

    def forward(self,x,g):
        valid=g['pair_mask']; b,j,m,d=x.shape
        job=masked_mean(x,valid,2)[:,:,None].expand_as(x)
        machine=masked_mean(x,valid,1)[:,None].expand_as(x)
        jcount=torch.log1p(valid.sum(2).float())[:,:,None,None].expand(b,j,m,1)
        mcount=torch.log1p(valid.sum(1).float())[:,None,:,None].expand(b,j,m,1)
        event,ecount=group_context(x.reshape(b,j*m,d),g['event_group'].reshape(b,j*m),valid.reshape(b,j*m))
        contexts=[self.projections[0](torch.cat((job,jcount),-1)),
                  self.projections[1](torch.cat((machine,mcount),-1)),
                  self.projections[2](torch.cat((event.reshape(b,j,m,d),ecount.reshape(b,j,m,1)),-1))]
        # Departure factors express a shared timetable, not finite vehicle capacity.
        # No event message on immediate links or on the no-TRG ablation.
        contexts[2]=contexts[2]*(g['event_group']>=0)[...,None]
        weights=torch.softmax(self.gate(torch.cat((x,*contexts),-1)),-1)
        aggregate=sum(weights[...,k,None]*v for k,v in enumerate(contexts))
        return self.norm(x+self.update(torch.cat((x,aggregate),-1)))*valid[...,None]


class TRGEncoder(nn.Module):
    """Encode the Timetable-Lifted Reachability Graph into action embeddings."""
    def __init__(self,d=32,heads=4,layers=2):
        super().__init__()
        if d % heads or min(d,heads,layers)<=0:
            raise ValueError('hidden_dim must be divisible by positive attention_heads')
        self.d,self.heads,self.width=d,heads,d//heads
        self.op_input=nn.Linear(TGRL_OP_DIM,d); self.ma_input=nn.Linear(TGRL_MA_DIM,d)
        self.pair_input=nn.Sequential(nn.Linear(TGRL_PAIR_DIM,d),nn.GELU(),nn.Linear(d,d))
        self.layers=nn.ModuleList([RelationLayer(d,heads) for _ in range(layers)])
        self.suffix_q,self.suffix_k,self.suffix_v=nn.Linear(d,d),nn.Linear(d,d),nn.Linear(d,d)
        self.action_input=nn.Linear(4*d,d)
        self.next_edge=nn.Sequential(nn.Linear(TGRL_NEXT_DIM,d),nn.GELU(),nn.Linear(d,d))
        self.next_q,self.next_k,self.next_v=nn.Linear(d,d),nn.Linear(d,d),nn.Linear(d,d)
        self.next_norm=nn.LayerNorm(d)
        self.factors=nn.ModuleList([EventFactorLayer(d) for _ in range(layers)])

    def forward(self,g):
        o=self.op_input(g['tgrl_op'])*g['op_mask'][...,None]
        m=self.ma_input(g['tgrl_ma'])
        for layer in self.layers: o,m=layer(o,m,g)
        b,n,d=o.shape; j=g['candidate'].size(1); nm=m.size(1)
        current=o.gather(1,g['candidate'][...,None].expand(-1,-1,d))
        successor=o.gather(1,g['successor'][...,None].expand(-1,-1,d))
        q=self.suffix_q(current).view(b,j,self.heads,self.width).transpose(1,2)
        k=self.suffix_k(o).view(b,n,self.heads,self.width).transpose(1,2)
        v=self.suffix_v(o).view(b,n,self.heads,self.width).transpose(1,2)
        owner=torch.arange(j,device=o.device)[None,:,None]
        pos=torch.arange(n,device=o.device)[None,None,:]
        mask=(g['job_id'][:,None]==owner)&(pos>=g['candidate'][...,None])&g['op_mask'][:,None]
        alpha=masked_softmax(q@k.transpose(-1,-2)/math.sqrt(self.width),mask[:,None],-1)
        suffix=(alpha@v).transpose(1,2).reshape(b,j,d)
        action=self.action_input(torch.cat((current[:,:,None].expand(-1,-1,nm,-1),
                                           m[:,None].expand(-1,j,-1,-1),
                                           suffix[:,:,None].expand(-1,-1,nm,-1),
                                           self.pair_input(g['tgrl_pair'])),-1))
        # Directed current-assignment -> successor-machine catchability relation.
        target=successor[:,:,None,None]+m[:,None,None]+self.next_edge(g['next_edge'])
        qn=self.next_q(action).view(b,j,nm,self.heads,self.width)
        kn=self.next_k(target).view(b,j,nm,nm,self.heads,self.width)
        vn=self.next_v(target).view(b,j,nm,nm,self.heads,self.width)
        scores=(qn[:,:,:,None]*kn).sum(-1)/math.sqrt(self.width)
        alpha=masked_softmax(scores,g['next_mask'][...,None],dim=3)
        next_message=(alpha[...,None]*vn).sum(3).reshape(b,j,nm,d)
        action=self.next_norm(action+next_message)*g['pair_mask'][...,None]
        for factor in self.factors: action=factor(action,g)
        pooled=torch.cat((masked_mean(o,g['op_mask'],1),m.mean(1),
                          masked_mean(action.flatten(1,2),g['pair_mask'].flatten(1),1)),-1)
        return action,pooled
