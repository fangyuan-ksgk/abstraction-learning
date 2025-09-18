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
    perplexity_increase_threshold: float,
    level_to_add: int = 1,
):
    """
    (TBD). remove threshold, just use sort order to decide place to add abstraction 
         - also consider default to rhythmic placeholder when tying etc. (to be considered)
    Analyzes a batch, identifies perplexity spikes based on the increase
    in perplexity, and inserts abstract placeholder tokens at those locations.
    """
    # 1. Ensure no abstract tokens exist yet
    abstract_mask = (batch_data.levels > 0)
    assert not abstract_mask.any(), "Cannot add placeholders; abstract tokens already exist."
    assert level_to_add > 0 and level_to_add < gat.L, f"level_to_add must be between 1 and {gat.L-1}."

    with torch.no_grad():
        # 2. Get per-token perplexity directly from the GAT model.
        # Note: gat() needs to return a tensor of shape [total_tokens].
        ppt = gat(batch_data)
        
    # 3. Detect spikes and insert placeholders for each sample
    new_batch = batch_data.clone()
    mask_token = gat.level_mask_tokens[level_to_add].item()

    for sample_idx in batch_data.indices:
        sample_mask_for_ts = (batch_data.sample_idx == sample_idx)
        sample_timestamps = batch_data.timestamps[sample_mask_for_ts]
        
        if len(sample_timestamps) <= 2:
            continue

        # Isolate the perplexities for the current sample.
        sample_ppt = ppt[sample_mask_for_ts]
        
        # Calculate the increase in perplexity
        ppt_increase = sample_ppt[1:] - sample_ppt[:-1]
        
        # Find locations where the increase exceeds the threshold
        # We look at ppt_increase[i] which corresponds to the change at sample_timestamps[i+1]
        spike_indices = (ppt_increase > perplexity_increase_threshold).nonzero(as_tuple=True)[0]
        
        if spike_indices.numel() > 0:
            # We insert the placeholder *before* the token that had the spike.
            # A spike at `spike_indices[k]` is for the token at `sample_timestamps[spike_indices[k] + 1]`.
            timestamps_to_add = sample_timestamps[spike_indices + 1]
            
            new_batch.insert_tokens(sample_idx, mask_token, level_to_add, timestamps_to_add)

    return new_batch





