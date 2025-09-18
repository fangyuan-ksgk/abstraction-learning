# utils & function for self-organizing reinforcemnet learning (GAT)
# ------------------------------------------------------------------------------------------------

# (Diminish Memory requires: dropping traj tokens in HierSeq)
# (Rythmic search requires: fixed interal addition of mask abstract tokens)
# (Adaptive search ideally requires: fixed-interval-addtion --> high-perplexity-detection --> selective addition of abstraction tokens)

# Question: 
# What's the minimal # of function that can implement above approaches?

import torch
from utils import HierSeq, repeat_hseq
from typing import Optional, List
from gat import GAT
from torch.distributions import Categorical
from collections import deque
import torch.nn.functional as F

# (Your other sorl.py functions like add_rhythmic_placeholders can remain for other experiments)

def add_rhythmic_placeholders(batch_data: HierSeq, level_mask_tokens: torch.Tensor, t_search: Optional[int] = None): 
    """
    Adds MASK_TOK placeholders at fixed, rhythmic intervals.
    If `t_search` is provided, it only adds placeholders up to that timestamp.
    Otherwise, it pads the entire sequence.
    """
    abstract_mask = (batch_data.levels > 0)
    assert not abstract_mask.any(), "Cannot add placeholders; abstract tokens already exist."

    for sample_idx in batch_data.indices: 
        sample_mask = (batch_data.sample_idx == sample_idx)
        sample_timestamps = batch_data.timestamps[sample_mask]
        start_ts, end_ts = sample_timestamps[0], sample_timestamps[-1]

        for l in range(1, batch_data.L): 
            interval = batch_data.K ** l
            abs_tok_ts = torch.arange(start_ts - 1, end_ts + 1, interval, device=start_ts.device)
            
            valid_ts_mask = (abs_tok_ts >= start_ts)

            if t_search is not None:
                valid_ts_mask &= (abs_tok_ts <= start_ts + t_search)
            
            abs_tok_ts_to_add = abs_tok_ts[valid_ts_mask]
            
            if abs_tok_ts_to_add.numel() > 0:
                mask_token_for_level = level_mask_tokens[l].item()
                batch_data.insert_tokens(sample_idx, mask_token_for_level, l, abs_tok_ts_to_add)
    
    return batch_data

def drop_traj_tokens(batch_data: HierSeq, t_keep: int): 
    """
    Keeps all abstract tokens but drops level-0 tokens, only keeping the
    `t_keep` most recent ones for each sample.
    """
    if t_keep <= 0:
        return batch_data

    l0_mask = (batch_data.levels == 0)
    
    cutoff_timestamps = torch.full((batch_data.batch_size,), -1, device=batch_data.device)
    for i, sample_idx in enumerate(batch_data.indices):
        sample_l0_timestamps = batch_data.timestamps[l0_mask & (batch_data.sample_idx == sample_idx)]
        if len(sample_l0_timestamps) > t_keep:
            cutoff_timestamps[i] = sample_l0_timestamps[-t_keep]

    sample_cutoff_ts = cutoff_timestamps[batch_data.sample_idx]
    keep_mask = ~l0_mask | (batch_data.timestamps >= sample_cutoff_ts)
    
    return batch_data.filter(keep_mask)


# Rollout generation || 
# ------------------------------------------------------------------------------------------------

# This, more specificlly, is generate in 'parallel denoised' mode, with 'rhythmic placeholder' abstraction tokens

def generate_rollout_data(gat: GAT, batch_data: HierSeq, n: int, temperature: float, t_search: Optional[int] = None): 
 
    # repeat
    repeat_batch = repeat_hseq(batch_data, n)

    # add placeholders
    repeat_batch = add_rhythmic_placeholders(repeat_batch, gat.level_mask_tokens, t_search)

    # generate rollout 
    repeat_batch = gat.generate(repeat_batch, parallel=True, temperature=temperature)

    return repeat_batch


# Abstract allocation based on perplexity spikes
# ------------------------------------------------------------------------------------------------
# - abstraction, like CoT is used to multi-hop, with the goal of reducing perplexity
# - spikes detectio is more natural as it doesn't break segment with 'consecutive decreasing perplexity'. 
# - this is analogous to our token growth work, which verifies the effectiveness of this spike detection based approach
# - for memory fading, this is also more suitable, as then it's like we are using a 'longer token' to replace things

def add_placeholders_at_spikes(
    gat: GAT,
    batch_data: HierSeq,
    abstract_budgets: torch.Tensor,
):
    """
    Analyzes a batch, identifies perplexity spikes (a decrease followed by an
    increase), and inserts a fixed budget of abstract placeholder tokens at the
    locations with the highest perplexity increase.
    """
    abstract_mask = (batch_data.levels > 0)
    assert not abstract_mask.any(), "Cannot add placeholders; abstract tokens already exist."

    with torch.no_grad():
        # This assumes gat(batch) returns per-token perplexity (or loss)
        # of shape [total_tokens]
        ppt = gat(batch_data)

    ppt_increase = ppt[1:] - ppt[:-1]
    
    # This function modifies batch_data in-place, which is different from
    # previous versions that returned a new_batch. We'll stick to the
    # logic from your notes.
    for add_level in range(1, batch_data.L):
        abstract_budget = abstract_budgets[add_level]
        if abstract_budget == 0:
            continue

        # A spike is a dip (or flat) followed by a rise.
        # drop_before_mask[i] is true if ppt_increase[i-1] was <= 0
        drop_before_mask = torch.cat([
            torch.tensor([False], device=ppt.device), 
            ppt_increase[:-1] <= 0
        ])
        increase_mask = ppt_increase > 0
        spike_mask = drop_before_mask & increase_mask

        # A spike involves 3 consecutive tokens (t-1, t, t+1).
        # Ensure they all belong to the same sample to avoid adding placeholders
        # at the boundary between two sequences in the batch.
        # The spike_mask has length T-1. We check sample boundaries for indices
        # i and i+1, which covers the main part of the spike.
        same_sample_mask = (batch_data.sample_idx[1:-1] == batch_data.sample_idx[2:])
        # spike_mask is T-1, same_sample_mask is T-2. We align them.
        spike_mask[1:] &= same_sample_mask
        
        spike_indices = torch.where(spike_mask)[0]
        if spike_indices.numel() == 0:
            continue
            
        # Order spikes by the magnitude of the perplexity increase
        spike_indices = spike_indices[torch.argsort(ppt_increase[spike_mask], descending=True)]
        
        # Select the top spikes up to the budget
        top_spike_indices = spike_indices[:abstract_budget]

        # Get the sample indices and timestamps for token insertion
        # A spike at spike_indices[k] means we insert a token before token [spike_indices[k] + 1]
        add_sample_idx = batch_data.sample_idx[top_spike_indices + 1]
        add_timestamps = batch_data.timestamps[top_spike_indices + 1]
        mask_token = gat.level_mask_tokens[add_level]

        batch_data.insert_tokens(add_sample_idx, mask_token, add_level, add_timestamps)

    return batch_data





