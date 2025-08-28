import torch 

from model import GAT 
from utils import remove_pad_tokens, HierSeq
import numpy as np


# Replicate HierSeq (n_copies)

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

    unique_original_indices = batch_data.sample_idx.unique(sorted=True)  # Get actual unique sample indices
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
    traj_mask = (repeat_batch.levels[1:] == 0) & (repeat_batch.timestamps[1:] > 0)
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
            next_mask = (repeat_batch.levels[1:] == level) & (repeat_batch.timestamps[1:] > 0)
            current_level_ppt = ppt[next_mask]
            current_level_idx = repeat_batch.sample_idx[1:][next_mask]

    return reward_lookups 








# Evaluation Gadget
# --------------------------------------------------------------------------------------------------------------------------
def eval_hseq(gat: GAT, batch_data: HierSeq, p_thres=4.16): 
    # remove_pad_tokens(batch_data)
    p_per_sample, critical_ts, cr_per_sample, ppt = gat(batch_data, evaluate=True, p_thres=p_thres) # per-sample avg. perplexity (different weight for each level's avg. perplexity)
    
    return p_per_sample, critical_ts, cr_per_sample, ppt