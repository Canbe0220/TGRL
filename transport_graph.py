"""Timetable-Lifted Reachability Graph (TRG).
"""
import math
import torch

# Encoder inputs are distinct from the environment's 6/4/4 raw features.
TGRL_OP_DIM, TGRL_MA_DIM, TGRL_PAIR_DIM, TGRL_NEXT_DIM = 8, 6, 8, 8
TGRL_PAIR_FEATURE_NAMES = (
    'processing_time',
    'travel_duration',
    'departure_wait',
    'earliest_dispatch_delay',
    'inter_factory_indicator',
    'relaxed_remaining_delivery',
    'delivery_bound_increment',
    'mean_successor_departure_wait',
)
# Channels disabled by the existing no-TRG ablation, in the compact schema.
TGRL_PAIR_TIMETABLE_CHANNELS = (2, 4, 5, 6, 7)


def gather_operations(x, indices):
    return x.gather(1, indices[..., None].expand(-1, -1, x.size(-1)))


def relaxed_completion_bound(proc, step, last, first_finish, compatible, availability,
                   distance, factory, period):
    """Exact FIFO earliest-arrival DP on each job's relaxed time-dependent DAG.

    Returns [B,J,M]: delivered completion conditioned on each first machine.
    Complexity O(B*J*L*M^3); memory O(B*J*M^3), independent of horizon length.
    The first M axis retains ALL initial alternatives, not just a greedy path.
    """
    b, j, m = first_finish.shape
    eye = torch.eye(m, dtype=torch.bool, device=proc.device)[None, None]
    inf = float('inf')
    frontier = torch.where(eye & compatible[..., None], first_finish[..., None], inf)
    cross = factory[:, :, None] != factory[:, None, :]
    transit = distance[:, 1:m+1, 1:m+1]
    horizon = int((last - step).clamp_min(0).max().item())
    for hop in range(1, horizon + 1):
        index = torch.minimum(step + hop, last)
        next_proc = gather_operations(proc, index)
        dispatch = torch.maximum(frontier[..., None], availability[:, None, None, None, :])
        p = period[:, None, None, None, None]
        departure = torch.where(cross[:, None, None], torch.ceil(dispatch / p) * p, dispatch)
        arrival = departure + transit[:, None, None] + next_proc[:, :, None, None, :]
        candidate = arrival.min(dim=-2).values.masked_fill(next_proc[:, :, None, :] <= 0, inf)
        frontier = torch.where((step + hop <= last)[:, :, None, None], candidate, frontier)
    delivered = (frontier + distance[:, None, None, 1:m+1, m+1]).min(dim=-1).values
    return delivered.masked_fill(~compatible, 0.0)


@torch.no_grad()
def build_trg(state, use_timetable=True):
    """Build observable TRG features without changing the environment.
    """
    ids = state.batch_idxes
    env_op_features = state.feat_opes_batch[ids].transpose(1, 2)
    proc = state.initial_proc_times[ids]
    b, n, m = proc.shape
    device = proc.device
    job_id = state.opes_appertain_batch[ids]
    starts = state.num_ope_biases_batch[ids]
    ends = state.end_ope_biases_batch[ids]
    counts = state.nums_ope_batch[ids]
    steps = torch.minimum(state.ope_step_batch[ids], ends)
    j = steps.size(1)
    now = state.time_batch[ids]
    scale = state.max_proc[ids]
    period = state.shuttle_period[ids]
    factory = state.mac_fac_mapping[ids]
    distance = state.mac_distance[ids]
    availability = state.machines_batch[ids, :, 1]
    last_machine = state.last_mac_batch[ids]
    finished = state.mask_job_finish_batch[ids]
    positions = torch.arange(n, device=device)[None, :].expand(b, -1)
    valid = positions < state.nums_opes_batch[ids, None]
    unfinished = valid & (env_op_features[..., 0] == 0)
    om_mask = (proc > 0) & unfinished[..., None]
    min_proc = proc.masked_fill(proc <= 0, float('inf')).min(-1).values
    min_proc = torch.where(valid, min_proc, 0.0)
    prefix = torch.cat((torch.zeros((b, 1), device=device), min_proc.cumsum(1)), 1)
    op_end = ends.gather(1, job_id)
    op_start = starts.gather(1, job_id)
    op_count = counts.gather(1, job_id).clamp_min(1)
    suffix_proc = prefix.gather(1, op_end + 1) - prefix[:, :-1]
    degree = (proc > 0).sum(-1).clamp_min(1)
    mean_proc = proc.sum(-1) / degree
    candidate_op = positions == steps.gather(1, job_id)
    tgrl_op = torch.stack((unfinished.float(), candidate_op.float(), degree.float()/m,
                      mean_proc/scale[:, None], suffix_proc/scale[:, None],
                      (positions-op_start).float()/op_count,
                      (env_op_features[..., 5]-now[:, None]).clamp_min(0)/scale[:, None],
                      env_op_features[..., 3]/op_count), -1) * unfinished[..., None]
    remaining_n = unfinished.sum(1).clamp_min(1)
    machine_degree = om_mask.sum(1)
    workload = ((proc / degree[..., None]) * om_mask).sum(1)
    tgrl_ma = torch.stack((machine_degree.float()/remaining_n[:, None],
                      (proc*om_mask).sum(1)/machine_degree.clamp_min(1)/scale[:, None],
                      (availability-now[:, None]).clamp_min(0)/scale[:, None],
                      state.mask_ma_procing_batch[ids].float(),
                      torch.log1p(workload/scale[:, None]),
                      (now[:, None]-availability).clamp_min(0)/scale[:, None]), -1)
    previous_index = (state.ope_step_batch[ids]-1).clamp_min(0)
    known_completion = env_op_features[..., 5] + env_op_features[..., 2]
    ready = known_completion.gather(1, previous_index)
    ready = torch.where(state.ope_step_batch[ids] > starts, ready, now[:, None])
    dispatch = torch.maximum(torch.maximum(now[:, None], ready)[..., None], availability[:, None])
    source_factory = factory.gather(1, last_machine.clamp_min(0))
    cross = (last_machine >= 0)[..., None] & (source_factory[..., None] != factory[:, None])
    p = period[:, None, None]
    departure = torch.where(cross, torch.ceil(dispatch/p)*p, dispatch)
    rows = torch.arange(b, device=device)[:, None, None]
    machines = torch.arange(1, m+1, device=device)[None, None]
    travel = distance[rows, (last_machine+1)[..., None], machines]
    current_proc = gather_operations(proc, steps)
    pair_mask = (current_proc > 0) & ~finished[..., None]
    eligible = (pair_mask & ~state.mask_job_procing_batch[ids, :, None] &
                ~state.mask_ma_procing_batch[ids, None, :])
    if not eligible.flatten(1).any(1).all():
        raise RuntimeError('Active state has no feasible action')
    wait = departure-dispatch
    finish = departure+travel+current_proc
    successor = torch.minimum(steps+1, ends)
    successor_proc = gather_operations(proc, successor)
    has_successor = (steps < ends) & ~finished
    next_mask = (pair_mask[..., None] & (successor_proc > 0)[:, :, None] &
                 has_successor[:, :, None, None])
    next_dispatch = torch.maximum(finish[..., None], availability[:, None, None, :])
    next_cross = (factory[:, :, None] != factory[:, None, :])[:, None].expand(-1, j, -1, -1)
    pp = period[:, None, None, None]
    next_departure = torch.where(next_cross, torch.ceil(next_dispatch/pp)*pp, next_dispatch)
    next_wait = next_departure-next_dispatch
    next_travel = distance[:, None, 1:m+1, 1:m+1].expand(-1, j, -1, -1)
    next_pt = successor_proc[:, :, None].expand(-1, -1, m, -1)
    next_finish = next_departure+next_travel+next_pt
    next_phase = 2*math.pi*torch.remainder(next_dispatch, pp)/pp
    ss = scale[:, None, None, None]
    next_edge = torch.stack((next_pt/ss, next_travel/ss, next_wait/ss,
                             (next_finish-finish[..., None])/ss, next_cross.float(),
                             next_wait/pp, torch.sin(next_phase), torch.cos(next_phase)), -1)
    next_edge *= next_mask[..., None]
    if use_timetable:
        suffix = relaxed_completion_bound(proc, steps, ends, finish, pair_mask, availability,
                                distance, factory, period)
    else:
        # Ablation: processing-only suffix, with the same representation size.
        tail = suffix_proc.gather(1, steps) - min_proc.gather(1, steps)
        suffix = finish + tail[..., None] + distance[:, None, 1:m+1, m+1]
        suffix = suffix.masked_fill(~pair_mask, 0.0)
    job_floor = suffix.masked_fill(~pair_mask, float('inf')).min(-1).values
    delivery = known_completion.gather(1, ends)+state.job_return_times[ids]
    job_floor = torch.where(finished, delivery, job_floor)
    global_floor = job_floor.max(1).values
    action_floor = torch.maximum(suffix, global_floor[:, None, None])
    prior = (state.makespan_batch[ids, None, None]-action_floor)/scale[:, None, None]
    prior = prior.masked_fill(~pair_mask, 0.0)
    next_degree = next_mask.sum(-1).clamp_min(1)
    mean_next_wait = (next_wait*next_mask).sum(-1)/next_degree
    scale_pair = scale[:, None, None]
    
    tgrl_pair = torch.stack((current_proc/scale_pair, travel/scale_pair, wait/scale_pair,
                            (dispatch-now[:, None, None])/scale_pair, cross.float(),
                            (suffix-finish)/scale_pair,
                            (action_floor-global_floor[:, None, None])/scale_pair,
                            mean_next_wait/scale_pair), -1)
    tgrl_pair *= pair_mask[..., None]
    event_group = torch.full((b,j,m), -1, dtype=torch.long, device=device)
    if use_timetable:
        # IDs are used only for equivalence classes, never as learned features.
        for i in range(b):
            keys = torch.stack((source_factory[i,:,None].expand(j,m).reshape(-1),
                                factory[i,None,:].expand(j,m).reshape(-1),
                                torch.round(departure[i]/period[i]).long().reshape(-1)), -1)
            _, inverse = torch.unique(keys, dim=0, return_inverse=True)
            event_group[i] = torch.where(cross[i] & pair_mask[i], inverse.reshape(j,m), -1)
    else:
        # This switch disables TRG-specific relations, not transport physics.
        tgrl_pair[..., list(TGRL_PAIR_TIMETABLE_CHANNELS)] = 0
        next_edge.zero_(); next_mask.zero_()
    prev_mask = unfinished & (positions > op_start)
    next_op_mask = unfinished & (positions < op_end)
    return dict(tgrl_op=tgrl_op, tgrl_ma=tgrl_ma, op_mask=unfinished, job_id=job_id,
                prev=(positions-1).clamp_min(0), next=(positions+1).clamp_max(n-1),
                prev_mask=prev_mask, next_op_mask=next_op_mask,
                om_mask=om_mask, proc_norm=proc/scale[:,None,None],
                candidate=steps, successor=successor, tgrl_pair=tgrl_pair,
                next_edge=next_edge, next_mask=next_mask,
                pair_mask=pair_mask, eligible=eligible, event_group=event_group,
                factory=factory, q_prior=prior)
