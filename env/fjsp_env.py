import sys
import gym
import torch

from dataclasses import dataclass
from env.load_data import load_fjs, nums_detec
import numpy as np
import random
import copy
from utils.my_utils import read_json, write_json


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None
    opes_mask: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_pair_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch:  torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    # Read-only topology and live resource state for timetable graph construction.
    initial_proc_times: torch.Tensor = None
    mac_fac_mapping: torch.Tensor = None
    mac_distance: torch.Tensor = None
    shuttle_period: torch.Tensor = None
    last_mac_batch: torch.Tensor = None
    machines_batch: torch.Tensor = None
    job_return_times: torch.Tensor = None
    num_ope_biases_batch: torch.Tensor = None
    nums_ope_batch: torch.Tensor = None
    max_proc: torch.Tensor = None
    makespan_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_pair_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_pair_batch = proc_pair_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''
    def __init__(self, case, env_paras, data_source='case'):
        '''
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        '''

        # load paras
        # static
        self.show_mode = env_paras["show_mode"]  # Result display mode (deprecated in the final experiment)
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.num_fac = env_paras["num_fac"]
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        # load instance
        num_data = 10  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        shuttle_periods = []
        if data_source=='case':  # Generate instances through generators
            for i in range(self.batch_size):
                lines.append(case.get_case(i)[0])  # Generate an instance and save it
                num_jobs, num_mas, num_fac, shuttle_period, num_opes = nums_detec(lines[i])
                shuttle_periods.append(shuttle_period)
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                num_jobs, num_mas, num_fac, shuttle_period, num_opes = nums_detec(lines[i])
                shuttle_periods.append(shuttle_period)
                self.num_opes = max(self.num_opes, num_opes)
        # A vectorized batch may have different operation counts, but J/M/F must agree.
        for line in lines:
            j, m, f, period, n = nums_detec(line)
            if (j, m, f) != (self.num_jobs, self.num_mas, self.num_fac):
                raise ValueError("Instance J/M/F does not match env_paras")
            if min(j, m, f, n) <= 0 or period <= 0:
                raise ValueError("Instance dimensions and shuttle period must be positive")
        # load feats
        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], self.num_mas, self.num_opes, device=self.device)
            for j in range(num_data):
                tensors[j].append(load_data[j])
        self.shuttle_period = torch.tensor(shuttle_periods, dtype=torch.float, device=self.device)

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        self.mac_fac_mapping = torch.stack(tensors[8], dim=0)
        self.mac_distance = torch.stack(tensors[9], dim=0)

        self.last_mac_batch = torch.full(size=(self.batch_size, self.num_jobs), fill_value=-1, dtype=torch.long, device=self.device)

        self.opes_mask = torch.zeros(self.batch_size, self.num_opes, self.num_opes, device=self.device)
        for i in range(self.batch_size):
            start_idx = self.num_ope_biases_batch[i]
            end_idx = self.end_ope_biases_batch[i]
            for j, k in zip(start_idx, end_idx):
                for t in range(j, k + 1):
                    self.opes_mask[i, t, t:k+1] = 1


        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size, device=self.device)  # Current time of the environment
        self.N = torch.zeros(self.batch_size, device=self.device).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes), device=self.device)
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas), device=self.device)

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)

        first_opes_idx = self.num_ope_biases_batch
        first_opes_adj = self.ope_ma_adj_batch.gather(
            1, first_opes_idx.unsqueeze(-1).expand(-1, -1, self.num_mas)
        )
        init_trans_times_feat = self.mac_distance[:, 0, 1:self.num_mas+1].unsqueeze(1) # (batch, 1, num_mas)
        mean_init_trans = (first_opes_adj * init_trans_times_feat).sum(dim=-1).div(
            first_opes_adj.sum(dim=-1) + 1e-9
        )
        init_trans_opes = convert_feat_job_2_ope(mean_init_trans, self.opes_appertain_batch)

        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze(1) + init_trans_opes
        
        last_opes_idx = self.end_ope_biases_batch
        last_opes_adj = self.ope_ma_adj_batch.gather(
            1, last_opes_idx.unsqueeze(-1).expand(-1, -1, self.num_mas)
        )
        return_trans_times_feat = self.mac_distance[:, 1:self.num_mas+1, self.num_mas+1].unsqueeze(1)
        mean_return_trans = (last_opes_adj * return_trans_times_feat).sum(dim=-1).div(
            last_opes_adj.sum(dim=-1) + 1e-9
        )

        self.job_return_times = mean_return_trans.clone().detach().to(self.device)

        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        end_time_batch += self.job_return_times
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)


        proc_pair_batch = torch.zeros(size=(self.batch_size, num_jobs, num_mas, self.paras["pair_feat_dim"]), device=self.device)
        batch_indices = torch.arange(self.batch_size, device=self.device).unsqueeze(1).expand(-1, self.num_jobs)
        job_indices = torch.arange(self.num_jobs, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        start_opes = self.ope_step_batch[batch_indices, job_indices]

        init_mas = torch.arange(1, self.num_mas + 1, device=self.device).view(1, 1, -1)
        init_last_macs = (self.last_mac_batch + 1).unsqueeze(-1)
        init_trans_times = self.mac_distance[batch_indices.unsqueeze(-1), init_last_macs, init_mas].float()
        init_total_time = self.proc_times_batch[batch_indices, start_opes].float() + init_trans_times

        self.max_proc = (self.proc_times_batch.amax(dim=(1, 2)) +
                         self.mac_distance.amax(dim=(1, 2)) + self.shuttle_period).clamp_min(1.0)
        proc_pair_batch[..., 0] = self.ope_ma_adj_batch[batch_indices, start_opes]
        proc_pair_batch[..., 1] = init_total_time.div(self.max_proc[:, None, None])
        proc_pair_batch[..., 2] = proc_pair_batch[..., 1].div(torch.max(proc_pair_batch[..., 1], dim=-1, keepdim=True)[0] + 1e-8)
        proc_pair_batch[..., 3] = proc_pair_batch[..., 1].div(torch.max(proc_pair_batch[..., 1], dim=-2, keepdim=True)[0] + 1e-8)


        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1).float()
        feat_mas_batch[:, 1, :] = torch.sum(self.proc_times_batch, dim=1).div(feat_mas_batch[:, 0, :] + 1e-8)


        valid_opes = torch.arange(self.num_opes, device=self.device)[None, :] < self.nums_opes[:, None]
        feat_opes_batch *= valid_opes[:, None, :]
        self.initial_trans_opes = init_trans_opes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_pair_batch = proc_pair_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False, device=self.device)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False, device=self.device)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False, device=self.device)
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4), device=self.device)
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4), device=self.device)
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas), device=self.device)

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.done = self.done_batch.all()

        self.dispatch_times_batch = torch.full((self.batch_size, self.num_opes), -1.0, device=self.device)
        self.departure_times_batch = torch.full((self.batch_size, self.num_opes), -1.0, device=self.device)
        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_pair_batch=self.proc_pair_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              opes_mask = self.opes_mask,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_proc_pair_batch = copy.deepcopy(self.proc_pair_batch)
        self.old_state = copy.deepcopy(self.state)
        self.old_last_mac_batch = copy.deepcopy(self.last_mac_batch)
        self.old_job_return_times = copy.deepcopy(self.job_return_times)
        self._refresh_pair_features()
        self._sync_state()

    def step(self, actions):
        '''
        Environment transition function
        '''
        if actions.shape != (3, self.batch_idxes.numel()):
            raise ValueError("actions must be [3, number of active instances]")
        actions = actions.to(device=self.device, dtype=torch.long)
        if self.batch_idxes.numel() == 0:
            raise RuntimeError("step called after all instances are done; call reset")
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        if ((jobs < 0) | (jobs >= self.num_jobs) | (mas < 0) | (mas >= self.num_mas)
                | (opes < 0) | (opes >= self.num_opes)).any():
            raise ValueError("Action index is out of range")
        b = self.batch_idxes
        legal = ((opes == self.ope_step_batch[b, jobs]) &
                 ~self.mask_job_procing_batch[b, jobs] & ~self.mask_job_finish_batch[b, jobs] &
                 ~self.mask_ma_procing_batch[b, mas] & self.ope_ma_adj_batch[b, opes, mas].bool())
        if not legal.all():
            raise ValueError("Infeasible operation-machine dispatch")
        self.N[self.batch_idxes] += 1
        self.dispatch_times_batch[self.batch_idxes, opes] = self.time[self.batch_idxes]

        last_macs = self.last_mac_batch[self.batch_idxes, jobs]
        trans_times = self.mac_distance[self.batch_idxes, last_macs + 1, mas + 1].float()
        departure_times = self.time[self.batch_idxes].clone()
        valid_mask = last_macs >= 0

        if valid_mask.any():
            idx_valid = self.batch_idxes[valid_mask]
            last_mac_ids = last_macs[valid_mask]
            curr_mac_ids = mas[valid_mask]
            
            last_facs = self.mac_fac_mapping[idx_valid, last_mac_ids]
            curr_facs = self.mac_fac_mapping[idx_valid, curr_mac_ids]
            
            cross_factory_mask = last_facs != curr_facs
            if cross_factory_mask.any():
                cross_factory_full_mask = valid_mask.clone()
                cross_factory_full_mask[valid_mask] = cross_factory_mask
                idx_cross = self.batch_idxes[cross_factory_full_mask]
                departure_times[cross_factory_full_mask] = (
                    torch.ceil(self.time[idx_cross] / self.shuttle_period[idx_cross]) 
                    * self.shuttle_period[idx_cross]
                )
        
        is_last_ope = (opes == self.end_ope_biases_batch[self.batch_idxes, jobs])
        if is_last_ope.any():
            idx_last = self.batch_idxes[is_last_ope]
            jobs_last = jobs[is_last_ope]
            curr_mac_idx = mas[is_last_ope] + 1
            end_node_idx = self.num_mas + 1
            
            return_dist = self.mac_distance[idx_last, curr_mac_idx, end_node_idx].float().to(self.device)
            self.job_return_times[idx_last, jobs_last] = return_dist

        self.departure_times_batch[self.batch_idxes, opes] = departure_times
        real_start_times = departure_times + trans_times
        self.last_mac_batch[self.batch_idxes, jobs] = mas

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64, device=self.device)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack((torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
                                                                        torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
                                                                        proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1, opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i]+1] -= 1

        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[self.batch_idxes, 5, opes] = real_start_times
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                            self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze(1)\
                         * un_scheduled  # estimate start time of unscheduled opes
        untouched = self.feat_opes_batch[self.batch_idxes, 0, :].gather(
            1, self.num_ope_biases_batch[self.batch_idxes]) == 0
        initial_trans = self.initial_trans_opes[self.batch_idxes] * convert_feat_job_2_ope(
            untouched.float(), self.opes_appertain_batch[self.batch_idxes])
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times + initial_trans
        valid = torch.arange(self.num_opes, device=self.device)[None, :] < self.nums_opes[self.batch_idxes, None]
        self.feat_opes_batch[self.batch_idxes] *= valid[:, None, :]
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        end_time_batch += self.job_return_times[self.batch_idxes, :]
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[self.batch_idxes, :])


        # Update partial schedule (state)
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0), device=self.device), mas), dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0), device=self.device)
        self.machines_batch[self.batch_idxes, mas, 1] = real_start_times + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch==self.end_ope_biases_batch+1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        max_val = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max_val
        self.reward_batch /= self.max_proc
        self.makespan_batch = max_val
        
        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while bool(((flag_trans_2_next_time == 0) & ~self.done_batch).any()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        self.batch_idxes = torch.arange(self.batch_size, device=self.device)[~self.done_batch]
        self._refresh_pair_features()
        self._sync_state()
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        busy = self.mask_ma_procing_batch & (a > self.time[:, None])
        c = torch.where(busy, a, torch.full_like(a, float('inf'))).min(dim=1).values
        if (flag_need_trans & ~torch.isfinite(c)).any():
            raise RuntimeError("Deadlock: no feasible action and no future machine completion")
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        self.machines_batch[:, :, 0][d] = 1

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = torch.nonzero(jobs >= 0, as_tuple=True)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

        self._refresh_pair_features()

    def _refresh_pair_features(self):
        """Recompute every candidate at the current time, including unscheduled jobs.

        """
        b = torch.arange(self.batch_size, device=self.device)[:, None]
        step = torch.minimum(self.ope_step_batch, self.end_ope_biases_batch)
        proc = self.proc_times_batch.gather(1, step[..., None].expand(-1, -1, self.num_mas))
        compat = self.ope_ma_adj_batch.gather(1, step[..., None].expand_as(proc)).bool()
        previous = (self.ope_step_batch - 1).clamp_min(0)
        ready = self.schedules_batch[:, :, 3].gather(1, previous)
        ready = torch.where(self.ope_step_batch > self.num_ope_biases_batch, ready, self.time[:, None])
        dispatch = torch.maximum(torch.maximum(self.time[:, None], ready)[:, :, None],
                                 self.machines_batch[:, None, :, 1])
        previous_fac = self.mac_fac_mapping.gather(1, self.last_mac_batch.clamp_min(0))
        cross = ((self.last_mac_batch >= 0)[:, :, None] &
                 (previous_fac[:, :, None] != self.mac_fac_mapping[:, None, :]))
        period = self.shuttle_period[:, None, None]
        depart = torch.where(cross, torch.ceil(dispatch / period) * period, dispatch)
        machines = torch.arange(1, self.num_mas + 1, device=self.device)[None, None, :]
        travel = self.mac_distance[b[:, :, None], (self.last_mac_batch + 1)[:, :, None], machines]
        duration = depart - dispatch + travel + proc
        active = compat & ~self.mask_job_finish_batch[:, :, None]
        self.proc_pair_batch[..., 0] = active.float()
        self.proc_pair_batch[..., 1] = torch.where(active, duration / self.max_proc[:, None, None], 0.0)
        cost = self.proc_pair_batch[..., 1]
        self.proc_pair_batch[..., 2] = cost / cost.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        self.proc_pair_batch[..., 3] = cost / cost.amax(dim=-2, keepdim=True).clamp_min(1e-8)
        unfinished = (self.feat_opes_batch[:, 0, :] == 0) & (
            torch.arange(self.num_opes, device=self.device)[None, :] < self.nums_opes[:, None])
        relation = self.ope_ma_adj_batch.bool() & unfinished[:, :, None]
        count = relation.sum(dim=1)
        self.feat_mas_batch[:, 0, :] = count.float()
        self.feat_mas_batch[:, 1, :] = (self.proc_times_batch * relation).sum(dim=1) / count.clamp_min(1)
        self.feat_mas_batch[:, 2, :] = self.mask_ma_procing_batch.float()
        self.feat_mas_batch[:, 3, :] = (self.time[:, None] - self.machines_batch[:, :, 1]).clamp_min(0)

    def _sync_state(self):
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch,
                          self.proc_pair_batch, self.proc_times_batch, self.ope_ma_adj_batch,
                          self.mask_job_procing_batch, self.mask_job_finish_batch,
                          self.mask_ma_procing_batch, self.ope_step_batch, self.time)
        self.state.initial_proc_times = self.old_proc_times_batch
        for name in ('mac_fac_mapping', 'mac_distance', 'shuttle_period', 'last_mac_batch',
                     'machines_batch', 'job_return_times', 'num_ope_biases_batch',
                     'nums_ope_batch', 'max_proc', 'makespan_batch'):
            setattr(self.state, name, getattr(self, name))

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.proc_pair_batch = copy.deepcopy(self.old_proc_pair_batch)
        self.state = copy.deepcopy(self.old_state)
        self.last_mac_batch = copy.deepcopy(self.old_last_mac_batch)
        self.job_return_times = copy.deepcopy(self.old_job_return_times)

        self.batch_idxes = torch.arange(self.batch_size, device=self.device)
        self.time = torch.zeros(self.batch_size, device=self.device)
        self.N = torch.zeros(self.batch_size, device=self.device)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False, device=self.device)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False, device=self.device)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False, device=self.device)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4), device=self.device)
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4), device=self.device)
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas), device=self.device)

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()
        self.dispatch_times_batch = torch.full((self.batch_size, self.num_opes), -1.0, device=self.device)
        self.departure_times_batch = torch.full((self.batch_size, self.num_opes), -1.0, device=self.device)
        self._refresh_pair_features()
        self._sync_state()
        return self.state

    def render(self, mode='human'):
        '''
        Deprecated in the final experiment
        '''
        if self.show_mode == 'draw':
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            num_jobs = self.num_jobs
            num_mas = self.num_mas
            print(sys.argv[0])
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                fig = plt.figure(figsize=(10, 6))
                fig.canvas.manager.set_window_title('Visual_gantt')
                axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('Machine {0}'.format(i + 1))
                    y_ticks_loc.append(i)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "job {0}".format(j + 1)
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                axes.cla()
                axes.set_title(u'FJSP Schedule')
                axes.grid(linestyle='-.', color='gray', alpha=0.2)
                axes.set_xlabel('Time')
                axes.set_ylabel('Machine')
                axes.set_yticks(y_ticks_loc)
                axes.set_yticklabels(y_ticks)
                axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                axes.set_ybound(-0.5, num_mas - 0.5)
                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    id_machine = schedules[id_ope][1]
                    axes.barh(id_machine,
                             0.2,
                             left=schedules[id_ope][2],
                             color='#b2b2b2',
                             height=0.5)
                    axes.barh(id_machine,
                             schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                             left=schedules[id_ope][2]+0.2,
                             color=color[idx_job],
                             height=0.5)
                plt.show()
        return

    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        """Audit processing, reservation, precedence, both transport modes and delivery.

        Unlike the original checker, this checks EVERY operation (including the
        last operation on each machine), and the whole [dispatch, completion)
        reservation interval rather than only its processing interval.
        """
        tol = 1e-5
        schedule = self.schedules_batch.detach().cpu()
        dispatch = self.dispatch_times_batch.detach().cpu()
        departure = self.departure_times_batch.detach().cpu()
        proc = self.old_proc_times_batch.detach().cpu()
        distance = self.mac_distance.detach().cpu()
        factory = self.mac_fac_mapping.detach().cpu()
        starts = self.num_ope_biases_batch.detach().cpu()
        ends = self.end_ope_biases_batch.detach().cpu()
        errors = []
        for b in range(self.batch_size):
            reservations = [[] for _ in range(self.num_mas)]
            deliveries = []
            period = float(self.shuttle_period[b])
            for j in range(self.num_jobs):
                last_m, last_end = -1, 0.0
                for o in range(int(starts[b, j]), int(ends[b, j]) + 1):
                    status, machine, start, finish = schedule[b, o].tolist()
                    m, d, dep = int(machine), float(dispatch[b, o]), float(departure[b, o])
                    if status != 1 or not 0 <= m < self.num_mas:
                        errors.append((b, o, 'unscheduled or invalid machine'))
                        continue
                    if proc[b, o, m] <= 0 or abs(finish - start - float(proc[b, o, m])) > tol:
                        errors.append((b, o, 'processing time'))
                    if d < last_end - tol:
                        errors.append((b, o, 'dispatch precedes predecessor completion'))
                    cross = last_m >= 0 and factory[b, last_m] != factory[b, m]
                    expected_dep = np.ceil(d / period) * period if cross else d
                    if abs(dep - expected_dep) > tol or abs(start - dep - float(distance[b, last_m + 1, m + 1])) > tol:
                        errors.append((b, o, 'transport/departure'))
                    reservations[m].append((d, finish, o))
                    last_m, last_end = m, finish
                if last_m >= 0:
                    deliveries.append(last_end + float(distance[b, last_m + 1, self.num_mas + 1]))
            for intervals in reservations:
                intervals.sort()
                for a, z in zip(intervals, intervals[1:]):
                    if a[1] > z[0] + tol:
                        errors.append((b, z[2], 'reservation overlap'))
            if deliveries and abs(max(deliveries) - float(self.makespan_batch[b])) > tol:
                errors.append((b, -1, 'delivery makespan'))
        self.validation_errors = errors
        return not errors, self.schedules_batch

    def close(self):
        pass
