import math
import torch 

from model import GAT 
from utils import HierSeq, pad_abstract_tokens, extend_abstract_tokens, get_unique_ordered, append_hseq
from utils import pad_answer_abstract_tokens, slice_query_hseq

import numpy as np

from dataclasses import dataclass
from typing import Optional


# Get Batch functional  | for training, no timestamp for answer is required, for evaluation we need it
# --------------------------------------------------------------------------------------------------------------------------

def get_batch(sequences: list, lengths: list, max_length: int, L: int, K: int):
    rand_idx = np.random.randint(0, len(sequences))
    batch = []
    sample_indices = []
    curr_len = 0

    for idx in range(rand_idx, len(sequences) + rand_idx):
        idx = idx % len(sequences)
        seq = sequences[idx]
        l = lengths[idx]
        if curr_len + l > max_length:
            break
        batch.append(([seq] + [[] for _ in range(1, L)], None))
        sample_indices.append(idx)
        curr_len += l

    batch_data = HierSeq.from_hierarchical_data(batch, sample_indices=sample_indices, K=K, L=L)
    return batch_data


# Buffer object
# --------------------------------------------------------------------------------------------------------------------------
from collections import defaultdict

class Buffer: 
    def __init__(self, size: int): 
        self.size = size
        self.record = defaultdict(list)
    
    def update(self, hseq: HierSeq): 
        h_seqs, h_timestamps = hseq.to_hierarchical_data()
        for i, idx in enumerate(hseq.indices): 
            if hseq.idx_map is not None: 
                i = hseq.idx_map[idx] # original index in dataset
            self.record[i].append(h_seqs[i][1:]) # record abstract tokens only
            self.record[i].append(h_timestamps[i][1:])


# Relevant function for SoRL training for GAT | Essentially a combination of GRPO & SSl - abstraction trained with RL, trajectory trained with SSL
# --------------------------------------------------------------------------------------------------------------------------

def repeat_hseq(batch_data: HierSeq, n_copies: int): 

    original_batch_size = batch_data.batch_size

    replicated_tokens = batch_data.tokens.repeat(n_copies)
    replicated_levels = batch_data.levels.repeat(n_copies)  
    replicated_timestamps = batch_data.timestamps.repeat(n_copies)

    tokens_per_sample = torch.bincount(batch_data.sample_idx)[batch_data.indices]  # [num_tokens_sample_0, num_tokens_sample_1, ...]
    repeats = tokens_per_sample.repeat(n_copies) 
    new_sample_idx = torch.repeat_interleave(
        torch.arange(original_batch_size * n_copies),
        repeats
    )

    unique_original_indices = batch_data.indices
    # unique_original_indices = batch_data.sample_idx.unique(sorted=True)  # Get actual unique sample indices
    idx_map = unique_original_indices.repeat(n_copies)  # Map each new sample back to its original

    replicated_batch_data = HierSeq(
        tokens=replicated_tokens,
        levels=replicated_levels,
        timestamps=replicated_timestamps,
        sample_idx=new_sample_idx,
        batch_size=original_batch_size * n_copies,
        K=batch_data.K,
        L=batch_data.L,
        idx_map=idx_map
    )

    return replicated_batch_data


def select_hseq(repeat_batch: HierSeq, select_mask: torch.Tensor): 
    batch_size = repeat_batch.batch_size

    selected_tokens = repeat_batch.tokens[select_mask]
    selected_levels = repeat_batch.levels[select_mask]
    selected_timestamps = repeat_batch.timestamps[select_mask]

    selected_sample_idx = repeat_batch.idx_map[repeat_batch.sample_idx[select_mask]]

    indices = get_unique_ordered(selected_sample_idx)
    assert batch_size % len(indices) == 0, f"Batch Size {batch_size} must be divisible by the number of selected samples {len(indices)}"

    selected_batch_size = len(indices)

    selected_batch_data = HierSeq(
        tokens=selected_tokens,
        levels=selected_levels,
        timestamps=selected_timestamps,
        sample_idx=selected_sample_idx,
        batch_size=selected_batch_size,
        K=repeat_batch.K,
        L=repeat_batch.L
    )
    return selected_batch_data





# Surrogate loss computation utils 
# --------------------------------------------------------------------------------------------------------------------------

def compute_per_sample_rewards(level_ppt: torch.Tensor, level_idx: torch.Tensor): 
    
    unique_samples, inverse = torch.unique(level_idx, return_inverse=True)

    n_unique = len(unique_samples)
    sums = torch.zeros(n_unique).scatter_add_(0, inverse, level_ppt)
    counts = torch.bincount(inverse, minlength=n_unique).float()
    averages = - sums / counts

    lookup = {s.item(): avg for s, avg in zip(unique_samples, averages)}
    return lookup 

# Cascaded Hierachical Reward Computation
def compute_hierarchical_rewards(ppt: torch.Tensor, repeat_batch: HierSeq) -> dict: 
    """ 
    abstraction level l is rewarded when it improves the perplexity of level l-1 tokens
    """
    traj_mask = (repeat_batch.levels[1:] == 0) & (repeat_batch.timestamps[1:] > 1)
    traj_ppt = ppt[traj_mask]
    traj_idx = repeat_batch.sample_idx[1:][traj_mask]

    reward_lookups = {}

    reward_lookups[0] = None

    current_level_ppt = traj_ppt
    current_level_idx = traj_idx

    for level in range(1, repeat_batch.L):
        # Compute per-sample average of current level
        sample_rewards = compute_per_sample_rewards(current_level_ppt, current_level_idx)
        reward_lookups[level] = sample_rewards
        
        # Prepare for next level (if exists)
        if level < repeat_batch.L - 1:
            next_mask = (repeat_batch.levels[1:] == level) & (repeat_batch.timestamps[1:] > 1)
            current_level_ppt = ppt[next_mask]
            current_level_idx = repeat_batch.sample_idx[1:][next_mask]

    return reward_lookups 


def compute_grouped_advantage(values: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped advantage using indices to group values tensor"""
    
    unique, inverse = torch.unique(indices, return_inverse=True)
    n = len(unique)

    means = torch.zeros(n).scatter_add_(0, inverse, values) / torch.bincount(inverse).float()
    vars = torch.zeros(n).scatter_add_(0, inverse, values**2) / torch.bincount(inverse).float() - means**2
    stds = torch.sqrt(torch.clamp(vars, min=0))

    advantages = (values - means[inverse]) / (stds[inverse] + 1e-4)

    return advantages

def compute_grouped_mean(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor: 
    """Grouped mean using indices to group values tensor"""
    unique, inverse = torch.unique(indices, return_inverse=True)
    n = len(unique)
    means = torch.zeros(n).scatter_add_(0, inverse, values) / torch.bincount(inverse).float()
    return means

def compute_grouped_max(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor: 
    """Grouped max using indices to group values tensor"""
    unique, inverse = torch.unique(indices, return_inverse=True)
    n = len(unique)
    maxs = torch.zeros(n).scatter_add_(0, inverse, values)
    return maxs


# (TBD). Implement a parameterized version of "reluctant argmax operation" - favors stability over changes 
# (TBD). Report 'advantage over greedy choice' value, too
def compute_grouped_max_mask(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:

    unique_groups, inverse = torch.unique(indices, return_inverse=True)
    num_groups = len(unique_groups)
    device = values.device

    max_values = torch.full((num_groups,), -float('inf'), device=device)
    max_values.scatter_reduce_(0, inverse, values, reduce='amax', include_self=True)

    rollout_advantages = torch.zeros(len(indices))

    is_max = (values == max_values[inverse])

    candidate_indices = torch.where(
        is_max, 
        torch.arange(len(values), device=device), 
        len(values) + 1
    )
    
    min_indices = torch.full((num_groups,), len(values) + 1, device=device, dtype=torch.long)
    min_indices.scatter_reduce_(0, inverse, candidate_indices, reduce='amin', include_self=True)

    final_mask = torch.zeros_like(values, dtype=torch.bool)
    valid_indices = min_indices[min_indices <= len(values)]
    final_mask[valid_indices] = True

    rollout_advantages[valid_indices] = values[valid_indices] - max_values[inverse[valid_indices]]

    return final_mask, rollout_advantages
    

def compute_weak_group_argmax_mask(means: torch.Tensor, orig_idx: torch.Tensor, indices: torch.Tensor, switch_abs_ppl_threshold: float = 0.1): 
    weak_argmax_mask = torch.zeros(len(orig_idx), dtype=torch.bool)
    rollout_advantages = torch.zeros(len(orig_idx))

    for idx in orig_idx:
        sample_mask = (orig_idx == idx)
        assert (indices[sample_mask] == indices[sample_mask].sort().values).all(), "First group is NOT the first appearance"
        
        # Pick the best rollout that satisfies the threshold condition
        rollout_ppl = means[sample_mask]
        greedy_ppl = rollout_ppl[0]
        rollout_advantages[sample_mask] = greedy_ppl - rollout_ppl

        mask = (rollout_ppl <= greedy_ppl - switch_abs_ppl_threshold)
        mask[0] = True

        effective_ppl = torch.where(mask, rollout_ppl, torch.tensor(float('inf')))
        abs_idx = torch.argmin(effective_ppl)
        abs_idx = torch.where(sample_mask)[0][abs_idx]

        weak_argmax_mask[abs_idx] = True
        
    return weak_argmax_mask, rollout_advantages


def compute_grouped_argmax(values: torch.Tensor, indices: torch.Tensor, idx_map: torch.Tensor): 

    # per-current-group mean (current indices)
    unique_indices, inverse = torch.unique(indices, return_inverse=True)
    n = len(unique_indices)
    means = torch.zeros(n).scatter_add_(0, inverse, values) / torch.bincount(inverse).float()

    # per-original-group argmax 
    orig_idx = idx_map[unique_indices]
    max_mask, rollout_advantages = compute_grouped_max_mask(means, orig_idx)
    argmax_indices = unique_indices[max_mask]

    # returned indices are in current indices space
    return argmax_indices, rollout_advantages[max_mask]


def compute_grouped_weak_argmax(values: torch.Tensor, indices: torch.Tensor, idx_map: torch.Tensor, switch_abs_ppl_threshold: float = 0.1): 

    # per-current-group mean (current indices)
    unique_indices, inverse = torch.unique(indices, return_inverse=True)
    n = len(unique_indices)
    means = torch.zeros(n).scatter_add_(0, inverse, values) / torch.bincount(inverse).float()

    # per-original-group argmax 
    orig_idx = idx_map[unique_indices]
    max_mask, rollout_advantages = compute_weak_group_argmax_mask(means, orig_idx, unique_indices, switch_abs_ppl_threshold)
    argmax_indices = unique_indices[max_mask]

    # returned indices are in current indices space
    return argmax_indices, rollout_advantages[max_mask]


# Per-abstract-token reward & advantage computation 
# --------------------------------------------------------------------------------------------------------------------------
def compute_abstract_token_rewards(repeat_batch: HierSeq, ppt: torch.Tensor) -> dict:
    
    lookup = {0: None}
    for level in range(1, repeat_batch.L):
        lookup[level] = {}
        abs_mask = (repeat_batch.levels[1:] == level)
        abs_level_sample = repeat_batch.sample_idx[1:][abs_mask]

        for sample_idx in abs_level_sample:

            sample_abs_mask = abs_mask & (repeat_batch.sample_idx[1:] == sample_idx)
            abs_ts = repeat_batch.timestamps[1:][sample_abs_mask]

            abs_affect_mask = (repeat_batch.timestamps[1:] > abs_ts[0]) & (repeat_batch.timestamps[1:] <= abs_ts[-1] + repeat_batch.K**level)
            traj_mask = (repeat_batch.levels[1:] == level - 1) & (repeat_batch.timestamps[1:] > 1) & (repeat_batch.sample_idx[1:] == sample_idx)
            traj_abs_mask = (traj_mask) & (abs_affect_mask)

            traj_abs_ppt = ppt[traj_abs_mask]
            traj_abs_idx = (repeat_batch.timestamps[1:][traj_abs_mask] - abs_ts[0] - 1) // repeat_batch.K**level

            means = compute_grouped_mean(traj_abs_ppt, traj_abs_idx)
            
            lookup[level][sample_idx.item()] = {ts.item(): rew for ts, rew in zip(abs_ts, means)}

    return lookup 


def compute_token_advantage(rewards, orig_indices, timestamps):
    """
    Computes advantage for each token, grouped by original sample index and timestamp.
    Advantages are normalized per-group (mean=0, std=1).
    """
    grouping_tensor = torch.stack([orig_indices, timestamps], dim=1)
    unique_groups, inverse_indices = torch.unique(grouping_tensor, dim=0, return_inverse=True)
    num_groups = len(unique_groups)
    device = rewards.device

    counts = torch.bincount(inverse_indices, minlength=num_groups).float()
    means = torch.zeros(num_groups, device=device).scatter_add_(0, inverse_indices, rewards) / counts
    vars = torch.zeros(num_groups, device=device).scatter_add_(0, inverse_indices, rewards**2) / counts - means**2
    stds = torch.sqrt(torch.clamp(vars, min=1e-8)) # clamp for stability

    expanded_means = means[inverse_indices]
    expanded_stds = stds[inverse_indices]
    
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-5)
    return advantages



# Rollout generation function with reference model
# --------------------------------------------------------------------------------------------------------------------------

def generate_rollout_data(gat: GAT, batch_data: HierSeq, n: int, temperature: float, t_search: Optional[int] = None): 
 
    repeat_batch = repeat_hseq(batch_data, n)

    if t_search is not None: 
        extend_abstract_tokens(repeat_batch, t_search) 
    else: 
        pad_abstract_tokens(repeat_batch) 

    # generate rollout 
    repeat_batch = gat.generate(repeat_batch, parallel=True, temperature=temperature)

    return repeat_batch


# Sitch together temp=0.0 sampled HierSeq with temp>0.0 sampled HierSeq
# --------------------------------------------------------------------------------------------------------------------------

def concatenate_hseq(hseq1, hseq2):

    hseq1.sample_idx = torch.cat([hseq1.sample_idx, hseq2.sample_idx + hseq1.batch_size])
    hseq1.batch_size = hseq1.batch_size + hseq2.batch_size
    hseq1.idx_map = torch.cat([hseq1.idx_map, hseq2.idx_map])

    hseq1.tokens = torch.cat([hseq1.tokens, hseq2.tokens])
    hseq1.levels = torch.cat([hseq1.levels, hseq2.levels])
    hseq1.timestamps = torch.cat([hseq1.timestamps, hseq2.timestamps])

    return hseq1


# Query & Answer rollout functionals 
# --------------------------------------------------------------------------------------------------------------------------

def generate_query_rollout_data(gat: GAT, batch_data: HierSeq, n: int, temperature: float, answer_token_id: int):

    repeat_batch = repeat_hseq(batch_data, n)
    slice_query_hseq(repeat_batch, answer_token_id)

    pad_abstract_tokens(repeat_batch)
    repeat_batch = gat.generate(repeat_batch, parallel=True, temperature=temperature)

    return repeat_batch 


def generate_answer_rollout_data(gat: GAT, batch_data: HierSeq, answer_token_id: int, n: int = 1, temperature: float = 0.0):

    repeat_batch = repeat_hseq(batch_data, n)
    pad_answer_abstract_tokens(repeat_batch, answer_token_id)
    repeat_batch = gat.generate(repeat_batch, parallel=True, temperature=temperature)
    return repeat_batch


# Pick best abstraction from rollout data & Repeat it to original length
# --------------------------------------------------------------------------------------------------------------------------
from sanity import print_switch_abstraction_ratio

def select_best_abstraction(repeat_batch: HierSeq, ppt: torch.Tensor, duplicate: bool = True, switch_abs_ppl_threshold: float = 0.0) -> tuple[HierSeq, float]: 
    """Pick best abstraction for each sample & repeat to original length || prioritize first occurance of max values when equal"""

    traj_mask = (repeat_batch.levels[1:] == 0) & (repeat_batch.timestamps[1:] > 1)
    traj_idx = repeat_batch.sample_idx[1:][traj_mask]
    traj_ppl = ppt[traj_mask]

    # if switch_abs_ppl_threshold > 0.0: 
    argmax_indices, rollout_advantages = compute_grouped_weak_argmax(traj_ppl, traj_idx, repeat_batch.idx_map, switch_abs_ppl_threshold)
    # else: 
    #     argmax_indices, rollout_advantages = compute_grouped_argmax(traj_ppl, traj_idx, repeat_batch.idx_map)

    # (TBD. remove this gadget) Information logging for visibility
    switch_ratio = print_switch_abstraction_ratio(repeat_batch, argmax_indices, rollout_advantages)

    select_mask = torch.isin(repeat_batch.sample_idx, argmax_indices)
    select_batch = select_hseq(repeat_batch, select_mask)
    if duplicate: 
        select_batch = repeat_hseq(select_batch, repeat_batch.batch_size // select_batch.batch_size)

    return select_batch, switch_ratio # (TBD. remove switch_ratio, it's used for logging purpose only)


# 2-in-1 search function: generate rollout & select best abstraction & repeat to original length
# --------------------------------------------------------------------------------------------------------------------------

def sorl_search(gat: GAT, batch_data: HierSeq, n: int, temperature: float, t_search: Optional[int] = None): 
    """Explore, Evaluate, Select"""

    if t_search is not None and t_search == 0: 
        return repeat_hseq(batch_data, n)

    # explore
    repeat_batch = generate_rollout_data(gat, batch_data, n, temperature, t_search)

    # evaluate 
    ppt = gat(repeat_batch)

    # select
    select_batch, switch_ratio = select_best_abstraction(repeat_batch, ppt)

    return select_batch, switch_ratio

def sorl_search_v2(gat: GAT, batch_data: HierSeq, n: int, temperature: float, t_search: Optional[int] = None, switch_abs_ppl_threshold: float = 0.0): 
    """Explore, Evaluate, Select || Pinned greedy sample ver."""
    
    if t_search is not None and t_search == 0: 
        return repeat_hseq(batch_data, n), 0.0

    if n == 1: 
        return sorl_search(gat, batch_data, n, temperature, t_search)

    # explore
    assert n > 1, "n must be greater than 1"
    ref_batch = generate_rollout_data(gat, batch_data, 1, 0.0, t_search)
    explore_batch = generate_rollout_data(gat, batch_data, n-1, temperature, t_search)

    ref_batch = concatenate_hseq(ref_batch, explore_batch)

    # evaluate 
    ppt = gat(ref_batch)

    # select | include threshold for weak-argmax selection that retains greedy sample (for stability)
    select_batch, switch_ratio = select_best_abstraction(ref_batch, ppt, switch_abs_ppl_threshold=switch_abs_ppl_threshold)

    return select_batch, switch_ratio




from sanity import sanity_check_repeat_batch

def sorl_search_query(gat: GAT, batch_data: HierSeq, n: int, temperature: float, answer_token_id: int):
    """Select from query-based abstractions (only one) & add answer HierSeq back"""
    # (Potential Issue). It's likely that timestamp of answer token is not maintained in 'repeat_batch'

    repeat_batch = generate_query_rollout_data(gat, batch_data, n, temperature, answer_token_id)

    ppt = gat(repeat_batch)

    select_batch, _ = select_best_abstraction(repeat_batch, ppt, duplicate=False)

    # In-place answer HierSeq appending
    append_hseq(select_batch, batch_data)

    return select_batch


# (Tentative) GRPO loss computation functional
# --------------------------------------------------------------------------------------------------------------------------

def compute_grpo_loss(repeat_batch: HierSeq, ppt: torch.Tensor,
                      old_log_probs: list, ref_log_probs: list, 
                      epsilon: float = 0.2, beta: float = 0.1, 
                      per_token_reward: bool = False):
        

    # per-level sample_idx->reward lookup table | detach makes sense?
    if per_token_reward: 
        reward_lookups = compute_abstract_token_rewards(repeat_batch, ppt.detach())
    else: 
        reward_lookups = compute_hierarchical_rewards(ppt.detach(), repeat_batch)

    loss = torch.tensor(0.0, device=repeat_batch.tokens.device)

    for l in range(1, repeat_batch.L): 
        level_mask = (repeat_batch.levels[1:] == l) & (repeat_batch.timestamps[1:] > 1)
        new_level_ppt = ppt[level_mask] 
        old_level_ppt = old_log_probs[l] # loaded from rollout data

        if len(old_level_ppt) == 0: 
            continue 

        if per_token_reward: 
            pt_sample_idx = repeat_batch.sample_idx[1:][level_mask]
            pt_ts = repeat_batch.timestamps[1:][level_mask]

            pt_reward = torch.stack([reward_lookups[l][idx.item()][ts.item()] for idx, ts in zip(pt_sample_idx, pt_ts)])
            pt_orig_idx = torch.stack([repeat_batch.idx_map[idx.item()] for idx in pt_sample_idx])

            advantages = compute_token_advantage(pt_reward, pt_orig_idx, pt_ts)# Per-abstract-token reward & advantage computation
        else: 
            # Sequence level Reward / Advantage computation
            # ------------------------------------------------------------------------------------------------
            level_sample_idx = repeat_batch.sample_idx[1:][level_mask]
            sample_with_level_l = repeat_batch.indices[torch.isin(repeat_batch.indices, level_sample_idx)]

            sample_level_rewards = torch.stack([reward_lookups[l][idx.item()] for idx in sample_with_level_l])
            orig_idx = torch.stack([repeat_batch.idx_map[idx.item()] for idx in sample_with_level_l])

            sample_level_advantages = compute_grouped_advantage(sample_level_rewards, orig_idx)

            sample_to_advantage = {
                sample_idx.item(): adv
                for sample_idx, adv in zip(sample_with_level_l, sample_level_advantages)
            }

            advantages = torch.stack([sample_to_advantage[idx.item()] for idx in level_sample_idx])
            # ------------------------------------------------------------------------------------------------
        

        ratio = torch.exp(new_level_ppt - old_level_ppt)

        # Compute surrogate loss
        surrogate_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages)

        # Compute KL divergence between reference & new log_probs
        log_ratio = ref_log_probs[l] - new_level_ppt
        kl = torch.exp(log_ratio) - log_ratio - 1

        # Compute per-token loss
        per_token_loss = surrogate_loss - beta * kl

        loss_abs_l = - per_token_loss.mean()

        loss += loss_abs_l

    return loss  


# (Issue). We got near-zero GSPO loss
# (TBD). Verify granular reward's effect on GRPO, then extend GSPO with per-token reward
def compute_gspo_loss(repeat_batch: HierSeq, ppt: torch.Tensor,
                      old_log_probs: list, ref_log_probs: list, 
                      epsilon: float = 0.2, beta: float = 0.1):

    reward_lookups = compute_hierarchical_rewards(ppt.detach(), repeat_batch)

    loss = torch.tensor(0.0, device=repeat_batch.tokens.device)

    for l in range(1, repeat_batch.L): 

        level_mask = (repeat_batch.levels[1:] == l) & (repeat_batch.timestamps[1:] > 1)
        new_level_ppt = ppt[level_mask] 
        old_level_ppt = old_log_probs[l] # loaded from rollout data

        if len(old_level_ppt) == 0: 
            continue 

        # Compute level l reward for each sample with abstraction at level l
        pt_sample_idx = repeat_batch.sample_idx[1:][level_mask]
        
        level_sample_idx = repeat_batch.sample_idx[1:][level_mask]
        sample_with_level_l = repeat_batch.indices[torch.isin(repeat_batch.indices, level_sample_idx)]

        sample_level_rewards = torch.stack([reward_lookups[l][idx.item()] for idx in sample_with_level_l])
        orig_idx = torch.stack([repeat_batch.idx_map[idx.item()] for idx in sample_with_level_l])

        sample_level_advantages = compute_grouped_advantage(sample_level_rewards, orig_idx)

        sample_to_advantage = {
            sample_idx.item(): adv
            for sample_idx, adv in zip(sample_with_level_l, sample_level_advantages)
        }

        advantages = torch.stack([sample_to_advantage[idx.item()] for idx in level_sample_idx])

        # per-sample avg. log prob ratio
        per_sample_ratio = compute_grouped_mean(new_level_ppt - old_level_ppt, pt_sample_idx)
        per_sample_ratio = torch.exp(per_sample_ratio)

        sample_to_ratio = {
            sample_idx.item(): ratio.item() 
            for sample_idx, ratio in zip(sample_with_level_l, per_sample_ratio)
        }

        ratio = torch.tensor(
            [sample_to_ratio[idx.item()] for idx in pt_sample_idx],
            device=repeat_batch.tokens.device
        )

        # Compute surrogate loss
        surrogate_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages)

        # Compute KL divergence between reference & new log_probs
        log_ratio = ref_log_probs[l] - new_level_ppt
        kl = torch.exp(log_ratio) - log_ratio - 1

        # Compute per-token loss
        per_token_loss = surrogate_loss - beta * kl

        loss_abs_l = - per_token_loss.mean()

        loss += loss_abs_l

    return loss


def compute_search_loss(repeat_batch: HierSeq, ppt: torch.Tensor):
    """Simplest possible ver. of RL loss | just maximize the advantage"""
        
    reward_lookups = compute_abstract_token_rewards(repeat_batch, ppt.detach())

    loss = torch.tensor(0.0, device=repeat_batch.tokens.device)

    for l in range(1, repeat_batch.L): 
        level_mask = (repeat_batch.levels[1:] == l) & (repeat_batch.timestamps[1:] > 1)
        new_level_ppt = ppt[level_mask] 

        pt_sample_idx = repeat_batch.sample_idx[1:][level_mask]
        pt_ts = repeat_batch.timestamps[1:][level_mask]

        pt_reward = torch.stack([reward_lookups[l][idx.item()][ts.item()] for idx, ts in zip(pt_sample_idx, pt_ts)])
        pt_orig_idx = torch.stack([repeat_batch.idx_map[idx.item()] for idx in pt_sample_idx])

        advantages = compute_token_advantage(pt_reward, pt_orig_idx, pt_ts)  # Per-abstract-token reward & advantage computation

        # Compute per-token loss
        per_token_loss = advantages * new_level_ppt

        loss_abs_l = - per_token_loss.mean()

        loss += loss_abs_l

    return loss  


def compute_ssl_loss(repeat_batch: HierSeq, ppt: torch.Tensor): 

    traj_mask =(repeat_batch.levels[1:] == 0) & (repeat_batch.timestamps[1:] > 1)
    traj_ppt = ppt[traj_mask]

    ssl_loss = traj_ppt.mean()

    return ssl_loss


def compute_dynamic_loss(repeat_batch: HierSeq, ppt: torch.Tensor): 
    """Dynamic language modeling loss Wu et al. 2025 (https://arxiv.org/abs/2508.05629)"""

    traj_mask =(repeat_batch.levels[1:] == 0) & (repeat_batch.timestamps[1:] > 1)
    traj_ppt = ppt[traj_mask]

    # Dynamic language modeling loss
    dynamic_loss = torch.exp(- traj_ppt).detach() * traj_ppt
    return dynamic_loss.mean() 


def compute_abs_ssl_loss(repeat_batch, ppt, level):

    ab_mask =(repeat_batch.levels[1:] == level) & (repeat_batch.timestamps[1:] > 1)
    ab_ppt = ppt[ab_mask]
    
    if len(ab_ppt) == 0: 
        return torch.tensor(0.0, device=repeat_batch.tokens.device)

    ssl_loss = ab_ppt.mean()

    return ssl_loss

# Curriculum search step increment
# --------------------------------------------------------------------------------------------------------------------------
def compute_curriculum_t_increment(num_iterations: int, context_length: int, K: int, max_ts: Optional[int]=None,
                                   num_loops: int = 1): 
    
    # maximal data length
    if max_ts is None: 
        max_ts = context_length
    else: 
        max_ts = min(max_ts, context_length)

    curriculum_iterations = int(num_iterations // num_loops * 0.8)

    max_abs_ts = (max_ts-1) // K * K # last timestamp needs no abstraction

    t_increment = math.ceil(max_abs_ts / curriculum_iterations) # additional search time-step per iteration

    return t_increment, max_abs_ts


def curriculum_iter(t_search: int, t_delta: int, t_max: int): 
    t_search += t_delta
    return t_search if t_search <= t_max else 0

# Phase Annealing Scheduler 
#   We perform phase transition (inner_loop_num, switch_abs_ppl_threshold) using (switch_ratio) metric 
#   When meaning stabilizes (switch_ratio < threshold), we switch from commitment stage to search stage
# --------------------------------------------------------------------------------------------------------------------------

# Remark: once into search stage, the returning condition should NOT be the same threshold. Search stage has much higher swtich_abstraction_ratio
class PhaseScheduler: 
    def __init__(self, init_joint_steps: int, init_abs_switch_ppl_threshold: float): 
        self.init_joint_steps = init_joint_steps
        self.init_abs_switch_ppl_threshold = init_abs_switch_ppl_threshold

        self.joint_steps = self.init_joint_steps
        self.abs_switch_ppl_threshold = self.init_abs_switch_ppl_threshold
        self.phase = "commitment"

    def __call__(self) -> tuple[int, float]: 
        raise NotImplementedError("PhaseScheduler is not implemented")

# Quality Metric: 
# (Embedding deviation)
# --------------------------------------------------------------------------------------------------------------------------
def compute_mbe(Z, detach=False, epsilon=1e-7):     
    gram = torch.bmm(Z, Z.transpose(1,2))
    if detach: 
        gram_trace = torch.diagonal(gram.detach(), dim1=1, dim2=2).sum(dim=1)
    else:
        gram_trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    gram_sq = gram.pow(2).sum(dim=(1,2))
    return -torch.log((gram_sq + epsilon) / (gram_trace.pow(2) + epsilon))

    
def compute_emb_mbe(gat: GAT, l: int): 
    assert l < gat.L, f"Level {l} must be less than the number of levels {gat.L}"
    wte = gat.wtes[l].weight.detach()
    proj = gat.lm_heads[l].weight.detach()
    mbe_wte = compute_mbe(wte.unsqueeze(0), detach=True)[0]
    mbe_proj = compute_mbe(proj.unsqueeze(0), detach=True)[0]
    return mbe_wte, mbe_proj


def eval_generate_ppl(gat: GAT, batch_data: HierSeq, n: int, temperature: float = 0.0, t_search: Optional[int] = None): 

    repeat_batch = repeat_hseq(batch_data, n)

    if t_search is not None: 
        extend_abstract_tokens(repeat_batch, t_search) 
    else: 
        pad_abstract_tokens(repeat_batch) 

    gat.generate(repeat_batch, parallel=True, temperature=temperature)

    # Compute log probs for each sample
    ppt = gat(repeat_batch)

    # per sample ppl avg 
    traj_mask = (repeat_batch.timestamps[1:] > 1) & (repeat_batch.levels[1:] == 0)
    traj_idx = repeat_batch.sample_idx[1:][traj_mask]
    traj_ppt = ppt[traj_mask]
    traj_orig_idx = repeat_batch.idx_map[traj_idx]

    per_sample_ppl = compute_grouped_mean(traj_ppt, traj_orig_idx)

    return per_sample_ppl


def eval_search_improvement(gat: GAT, batch_data: HierSeq, n: int = 5, t_search: Optional[int] = None): 

    with torch.no_grad():
        greedy_ppl = eval_generate_ppl(gat, batch_data, 1, temperature=0.0, t_search=t_search)

        random_ppl = eval_generate_ppl(gat, batch_data, n, temperature=100.0, t_search=t_search)

        # compute prediction entropy

    improve_ppl_percentage = (random_ppl - greedy_ppl) / random_ppl # percentage of improvement in ppl

    return improve_ppl_percentage.mean() * 100

def compute_traj_ppl_per_sample(hseq: HierSeq, gat: GAT): 
    
    ppt = gat(hseq)

    # per-sample traj_ppl computation
    traj_mask = (hseq.levels[1:] == 0) & (hseq.timestamps[1:] > 1)
    traj_idx = hseq.sample_idx[1:][traj_mask]
    traj_ppl = ppt[traj_mask]

    return compute_grouped_mean(traj_ppl, traj_idx)


# Evaluation with abstraction search for Query tokens 
# --------------------------------------------------------------------------------------------------------------------------
def eval_ppl_with_search(hseq: HierSeq, gat: GAT, answer_token_id: int,
                        n: int = 4, temperature: float =1.0): 
    """Search best abstraction for query, then generate answer & evaluate on trajectory perplexity"""

    query_batch_data = sorl_search_query(gat, hseq, n=n, temperature=temperature, answer_token_id=answer_token_id)

    answer_batch_data = generate_answer_rollout_data(gat, query_batch_data, answer_token_id=answer_token_id, n=1, temperature=0.0)

    ppl_per_sample = compute_traj_ppl_per_sample(answer_batch_data, gat)

    return ppl_per_sample


# evaluate entropy & ppl deviation
# --------------------------------------------------------------------------------------------------------------------------
def eval_entropy_ppl_deviation(gat: GAT, repeat_batch: HierSeq): 
    """Evaluate (mean) model entropy and (mean) ppl deviation (related to reward)"""

    with torch.no_grad():
        ppt, entropy = gat(repeat_batch, return_entropy=True)

    entropy_per_level = {}
    ppl_per_level = {}
    ppl_deviation_per_level = {}

    for l in range(gat.L):
        
        traj_mask = (repeat_batch.timestamps[1:] > 1) & (repeat_batch.levels[1:] == l)

        traj_entropy = entropy[traj_mask]
        traj_ppt = ppt[traj_mask]

        entropy_per_level[l] = traj_entropy.mean() 
        ppl_per_level[l] = traj_ppt.mean()
        ppl_deviation_per_level[l] = abs(traj_ppt - ppl_per_level[l]).mean()

    return entropy_per_level, ppl_per_level, ppl_deviation_per_level



# Evaluation Gadget
# --------------------------------------------------------------------------------------------------------------------------
def observe_abstraction(batch_data: HierSeq, gat: GAT, t_search: Optional[int] = None, temperature: float = 0.0): 

    repeat_batch = repeat_hseq(batch_data, 1)

    if t_search is not None: 
        extend_abstract_tokens(repeat_batch, t_search) 
    else: 
        pad_abstract_tokens(repeat_batch) 

    gat.generate(repeat_batch, parallel=True, temperature=temperature)

    info_str = ""
    for idx in repeat_batch.indices: 
        for l in range(1, repeat_batch.L): 
            abs_mask = (repeat_batch.levels == l) & (repeat_batch.sample_idx == idx)
            orig_idx = repeat_batch.idx_map[idx].item()
            info_str += f"Sample {orig_idx} Abstract Tokens at level {l}: {repeat_batch.tokens[abs_mask].tolist()}\n"

    return info_str


# Experiment Configuration 
# --------------------------------------------------------------------------------------------------------------------------
from model import GATConfig


# Ver.3 SoRL (with GRPO / RL loss)
@dataclass
class SORLConfig: 
    # model configuration 
    gat_config: GATConfig

    # training config
    n_generations: int = 2
    temperature: float = 1.0
    num_iterations: int = 10
    joint_steps: int = 20
    context_length: int = 2048 
    learning_rate: float = 1e-3
    t_curriculum: bool = True
    log_interval: int = 100
    use_v2: bool = True # if True, use v2 search function
    switch_abs_ppl_threshold: float = 0.0 # if > 0.0, use weak-argmax selection that retains greedy sample (for stability)

    # dataset 
    dataset_name: str = "2body_2k"
    dataset_path: str = "dataset/nbody/2body_2k.bin"

    # validation (in-domain & out-of-domain)
    id_validate_dataset_name: str = "2body_100"
    id_validate_dataset_path: str = "dataset/nbody/2body_100.bin"

    ood_validate_dataset_name: str = "3body_100"
    ood_validate_dataset_path: str = "dataset/nbody/3body_100.bin"




# Ver.2 SoRL (with GRPO / RL loss)
@dataclass
class SORLv2Config: 
    # model configuration 
    gat_config: GATConfig

    # training config
    gspo: bool = False # if False, use GRPO
    n_generations: int = 4
    temperature: float = 1.0
    num_iterations: int = 2
    num_steps: int = 10 
    joint_steps: int = 10 
    max_length: int = 1024 
    learning_rate: float = 1e-3
    epsilon: float = 0.2
    beta: float = 0.1
    per_token_reward: bool = False
    t_search: Optional[int] = None
    num_loops: int = 1
    anneal_window_size: int = 200
    anneal_threshold: float = 0.1
    
    # dataset 
    dataset_name: str = "2body_2k"
    dataset_path: str = "dataset/nbody/2body_2k.bin"

    # validation (in-domain & out-of-domain)
    id_validate_dataset_name: str = "2body_100"
    id_validate_dataset_path: str = "dataset/nbody/2body_100.bin"

    ood_validate_dataset_name: str = "3body_100"
    ood_validate_dataset_path: str = "dataset/nbody/3body_100.bin"


