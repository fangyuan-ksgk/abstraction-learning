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


# Now the question is, how to build other mode rollouts? 





