# utils & function for self-organizing reinforcemnet learning (GAT)
# ------------------------------------------------------------------------------------------------

# (Diminish Memory requires: dropping traj tokens in HierSeq)
# (Rythmic search requires: fixed interal addition of mask abstract tokens)
# (Adaptive search ideally requires: fixed-interval-addtion --> high-perplexity-detection --> selective addition of abstraction tokens)

# Question: 
# What's the minimal # of function that can implement above approaches?

import torch
from utils import HierSeq, repeat_hseq, compute_grouped_weak_argmax, select_hseq, compute_switch_abstraction_ratio, concatenate_hseq
from typing import Optional, List
from gat import GAT
from torch.distributions import Categorical
from collections import deque
import torch.nn.functional as F
from math import ceil


# ------------------------------------------------------------------------------------------------
# Placeholder enables parallel generation of abstraction, speeds up search significantly
# Two mode of placeholder addition: 
# - rhythmic placeholder: fixed interval addition (external)
# - spike placeholder: adaptive addition based on perplexity spikes (internal)
# ------------------------------------------------------------------------------------------------

def add_rhythmic_placeholders(batch_data: HierSeq, level_mask_tokens: torch.Tensor, t_search: Optional[int] = None): 
    """
    Adds MASK_TOK placeholders at fixed, rhythmic intervals.
    If `t_search` is provided, it only adds placeholders up to that timestamp.
    Otherwise, it pads the entire sequence.
    """

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
                batch_data.insert_tokens(sample_idx, mask_token_for_level, l, abs_tok_ts_to_add, overwrite=False)
    
    return batch_data


def allocate_budget(spike_weights, abstract_budget):
    normalized_weights = spike_weights / spike_weights.sum()
    ideal_counts = normalized_weights * abstract_budget
    token_counts = torch.floor(ideal_counts).long()
    remainder = abstract_budget - token_counts.sum()

    if remainder > 0:
        residuals = ideal_counts - token_counts
        _, top_indices = torch.topk(residuals, int(remainder))
        token_counts[top_indices] += 1

    return token_counts 


def add_spike_placeholders(
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
        ppt = gat(batch_data)

    ppt_increase = ppt[1:] - ppt[:-1]
    
    for add_level in range(1, batch_data.L):
        abstract_budget = abstract_budgets[add_level]
        if abstract_budget == 0:
            continue

        # increase in perplexity within same sample
        spike_mask = ppt_increase > 0
        same_sample_mask = (batch_data.sample_idx[1:-1] == batch_data.sample_idx[2:])
        spike_mask &= same_sample_mask

        spike_indices = torch.where(spike_mask)[0]
        if spike_indices.numel() == 0:
            continue
            
        spike_weights = ppt_increase[spike_mask]
        token_counts = allocate_budget(spike_weights, abstract_budget)

        non_zero_mask = token_counts > 0
        final_counts = token_counts[non_zero_mask]
        final_indices = spike_indices[non_zero_mask]

        if final_counts.numel() == 0:
            continue

        add_timestamps = torch.repeat_interleave(batch_data.timestamps[final_indices + 1], final_counts)
        add_sample_idx = torch.repeat_interleave(batch_data.sample_idx[final_indices + 1], final_counts)
        mask_token = gat.level_mask_tokens[add_level]

        for idx, ts in zip(add_sample_idx, add_timestamps):
            batch_data.insert_tokens(idx, mask_token, add_level, ts, overwrite=False)

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


def pad_abstract_tokens(batch_data: HierSeq, 
                        gat: GAT,
                        t_search: Optional[int] = None,
                        use_spike_placeholders: bool = False,
                        abstract_budgets: Optional[torch.Tensor] = None,
                        use_rhythmic_placeholders: bool = False,
                        use_diminish_memory: bool = False,
                        t_keep: int = 0):
    """
    Add placeholder abstraction tokens to HierSeq 
    """

    if use_spike_placeholders:
        assert abstract_budgets is not None, "abstract_budgets must be provided for spike placeholders"
    if use_diminish_memory:
        assert t_keep > 0, "t_keep must be > 0 for diminish memory"

    # 1. Optionally add spike-based placeholders
    if use_spike_placeholders:
        batch_data = add_spike_placeholders(gat, batch_data, abstract_budgets)

    # 2. Optionally add rhythmic placeholders
    if use_rhythmic_placeholders:
        batch_data = add_rhythmic_placeholders(batch_data, gat.level_mask_tokens, t_search)

    # 3. Optionally diminish memory by dropping old trajectory tokens
    if use_diminish_memory:
        batch_data = drop_traj_tokens(batch_data, t_keep)

    return batch_data


def repad_abstract_tokens(batch_data: HierSeq, keep_ratio: float, level_mask_tokens: torch.Tensor):
    """
    For each sample, keeps the first `keep_prefix_n` abstract tokens and
    replaces the rest with their corresponding MASK placeholder tokens.
    """
    n_replace = 0 

    for sample_idx in batch_data.indices:

        sample_abstract_mask = (batch_data.sample_idx == sample_idx.item()) & (batch_data.levels > 0)
        abstract_indices = sample_abstract_mask.nonzero(as_tuple=True)[0]
        keep_n = ceil(len(abstract_indices) * keep_ratio)
        
        if len(abstract_indices) > keep_n:
            indices_to_replace = abstract_indices[keep_n:]
            levels_to_replace = batch_data.levels[indices_to_replace]
            mask_tokens_to_insert = level_mask_tokens.to(batch_data.device)[levels_to_replace]
            batch_data.tokens[indices_to_replace] = mask_tokens_to_insert
            n_replace += len(indices_to_replace) 

    return batch_data, n_replace


# Generate chunk by chunk, parallel within chunk (KV cache to be added)
# (TBD). Include KV-cache inside. 
# --------------------------------------------------------------------------------------------------------------------
def chunk_generate(model: GAT, batch: HierSeq, n_step: int, temperature: float = 0.0):

    model.generate(batch, parallel=True, temperature=temperature)

    ratio_schedule = torch.linspace(0.0, 1.0, n_step+1)[1:-1]
    for ratio in ratio_schedule: 
        batch, n_replace = repad_abstract_tokens(batch, ratio.item(), model.level_mask_tokens)
        if n_replace == 0: 
            break 
        model.generate(batch, parallel=True, temperature=0.0)

    return batch


# ------------------------------------------------------------------------------------------------
# Causal Abstraction Generation (one-by-one) is the only way to leverage proper attention to previous abstractions
# - tricks like rhythmic abstraction generation can be easily integraed into causal generation
# - ideally the spike-based abstraction addition can also be integrated into causal generation
# - that way we unify "inference / evaluation" with "rollout generation"
# - it's just quite complicated to integrate causal generation .... 
# - is there an easier way of doing this? how about segment-wise parallel generation?
# - or, better yet, how about just denoising style iterative refinement? 
# - The simplest way: 
# ------------------------------------------------------------------------------------------------



# Rollout generation || Combining above functionalities
# ------------------------------------------------------------------------------------------------

def generate_rollout_data(
    gat: GAT,
    batch_data: HierSeq,
    n: int,
    temperature: float,
    t_search: Optional[int] = None,
    use_spike_placeholders: bool = False,
    abstract_budgets: Optional[torch.Tensor] = None,
    use_rhythmic_placeholders: bool = False,
    use_diminish_memory: bool = False,
    t_keep: int = 0,
    chunk_generate_step: int = 1,
    ):
    """
    Generates rollout data by repeating a batch, optionally applying SoRL
    techniques, and then denoising to generate the final trajectories.
    The order of application is: diminish memory, spike placeholders, rhythmic placeholders.
    """
    repeat_batch = repeat_hseq(batch_data, n)

    repeat_batch = pad_abstract_tokens(repeat_batch, 
                                       gat, 
                                       t_search, 
                                       use_spike_placeholders, 
                                       abstract_budgets, 
                                       use_rhythmic_placeholders, 
                                       use_diminish_memory, 
                                       t_keep)

    rollout_batch = chunk_generate(gat, repeat_batch, chunk_generate_step, temperature)

    return rollout_batch


# Sitch together temp=0.0 sampled HierSeq with temp>0.0 sampled HierSeq
# --------------------------------------------------------------------------------------------------------------------------

# (To be checked)
def select_best_abstraction(repeat_batch: HierSeq, ppt: torch.Tensor, duplicate: bool = True, switch_abs_ppl_threshold: float = 0.0) -> tuple[HierSeq, float]: 
    """Pick best abstraction for each sample & repeat to original length || prioritize first occurance of max values when equal"""

    traj_mask = (repeat_batch.levels[1:] == 0) & (repeat_batch.timestamps[1:] > 1)
    traj_idx = repeat_batch.sample_idx[1:][traj_mask]
    traj_ppl = ppt[traj_mask]

    argmax_indices, rollout_advantages = compute_grouped_weak_argmax(traj_ppl, traj_idx, repeat_batch.idx_map, switch_abs_ppl_threshold)

    switch_ratio = compute_switch_abstraction_ratio(repeat_batch, argmax_indices, rollout_advantages) # to be removed

    select_mask = torch.isin(repeat_batch.sample_idx, argmax_indices)
    select_batch = select_hseq(repeat_batch, select_mask)
    if duplicate: 
        select_batch = repeat_hseq(select_batch, repeat_batch.batch_size // select_batch.batch_size)

    return select_batch, switch_ratio, rollout_advantages # (TBD. remove switch_ratio, rollout_advantages, theses are for logging purpose only)

# (To be checked)
def sorl_search(gat: GAT, batch_data: HierSeq, n: int, temperature: float, t_search: Optional[int] = None,
                use_spike_placeholders: bool = False,
                abstract_budgets: Optional[torch.Tensor] = None,
                use_rhythmic_placeholders: bool = False,
                use_diminish_memory: bool = False,
                t_keep: int = 0,
    ): 

    """Explore, Evaluate, Select || Pinned greedy sample ver."""
    
    if t_search is not None and t_search == 0: 
        return repeat_hseq(batch_data, n), 0.0, torch.tensor([0.0])

    # explore
    assert n > 1, "n must be greater than 1"
    ref_batch = generate_rollout_data(gat, batch_data, 1, 0.0, t_search, use_spike_placeholders, abstract_budgets, use_rhythmic_placeholders, use_diminish_memory, t_keep)
    explore_batch = generate_rollout_data(gat, batch_data, n-1, temperature, t_search, use_spike_placeholders, abstract_budgets, use_rhythmic_placeholders, use_diminish_memory, t_keep)

    ref_batch = concatenate_hseq(ref_batch, explore_batch)

    # evaluate 
    ppt = gat(ref_batch)

    # select | include threshold for weak-argmax selection that retains greedy sample (for stability)
    select_batch, switch_ratio, rollout_advantages = select_best_abstraction(ref_batch, ppt)

    return select_batch, switch_ratio, rollout_advantages