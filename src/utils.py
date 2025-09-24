import torch
from typing import Optional, Union

def infer_level(indices: torch.Tensor, vocab_sizes: torch.Tensor, pad_token: int):
    indices_expanded = indices.unsqueeze(-1)  # [batch_size, seq_len, 1]
    levels = (indices_expanded < vocab_sizes.cumsum(dim=0)).int().argmax(dim=-1)

    padding_mask = (indices == pad_token)
    final_levels = torch.where(padding_mask, -1, levels.long())
    return final_levels

# this produces 'abstract timestamps'
def infer_timestamp(levels: torch.Tensor, K: int, l: int = 1) -> torch.Tensor:
    """timestamp starts from 0"""
    is_level = (levels == l-1).long()  
    cumulative_counts = torch.cumsum(is_level, dim=-1)
    timestamps = (cumulative_counts - 1) // K
    timestamps.clamp_(min=0) # this assings the correct timestamp 
    return timestamps

# Rhythmic insertion mask calculation
def infer_rythmic_insertion_mask(levels: torch.Tensor, timestamps: torch.Tensor, K: int, l: int): 
    """When t_search = 1, we insert placeholder up to timestamp <= 0, or timestamp < t_search"""
    within_level_mask = (levels <= l)
    timestamps[~within_level_mask] = False 

    B = timestamps.size(0)
    is_end_of_groups = torch.cat([
        (timestamps[:, :-1] != timestamps[:, 1:]),
        torch.full((B, 1), True, device=timestamps.device)
    ], dim=1)

    is_valid_elements = [] 
    for timestamp in timestamps: 
        group_counts = torch.bincount(timestamp) # count consecutive value group size
        is_valid_group = group_counts >= K 
        is_valid_element = is_valid_group[timestamp] # timestamp starts from 0 makes this valid
        is_valid_elements.append(is_valid_element)
    is_valid_elements = torch.stack(is_valid_elements, dim=0)

    insert_mask = is_end_of_groups & is_valid_elements
    insert_mask[:, -1] = False 
    insert_mask = torch.roll(insert_mask.int(), shifts=1, dims=1)
    
    return insert_mask # insert before 'True' position

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

def infer_spike_insertion_mask(levels: torch.Tensor, ppt: torch.Tensor, l: int, abstract_budget: int): 
    level_masks = (levels[:, 1:] == l-1)

    ppt_increase = torch.zeros_like(ppt[:, 1:], device=ppt.device).float()
    for i, (sample_ppt, level_mask) in enumerate(zip(ppt, level_masks)): 
        sample_ppt = sample_ppt[level_mask]
        ppt_increase[i, level_mask[1:]] = sample_ppt[1:] - sample_ppt[:-1]

    spike_mask = (ppt_increase > 0) 
    spike_indices = torch.where(spike_mask)[0]

    insert_masks = torch.zeros_like(spike_mask).int()
    if spike_indices.numel() > 0: 
        
        spike_weights = ppt_increase[spike_mask]
        token_counts = allocate_budget(spike_weights, abstract_budget)

        non_zero_mask = token_counts > 0
        final_counts = token_counts[non_zero_mask].int()

        # - insert_mask (we can put integer value into the mask, too)
        update_mask = torch.zeros_like(insert_masks, dtype=torch.bool)
        update_mask[spike_mask] = non_zero_mask
        insert_masks[update_mask] = final_counts
    
    insert_masks = torch.cat([torch.zeros_like(spike_mask).int()[:, :2], insert_masks], dim=1) # operate on 'data'
    return insert_masks

def infer_valid_masks(timestamps: torch.Tensor, start_ts: Optional[torch.Tensor] = None, t_search: Optional[Union[int, torch.Tensor]] = None, end_ts: Optional[torch.Tensor] = None): 
    
    if start_ts is None and t_search is None: 
        return torch.ones_like(timestamps, dtype=torch.int)

    if start_ts is None: 
        start_ts = torch.zeros_like(timestamps[:, 0])
    if t_search is None: 
        t_search = 1e9

    if end_ts is None: 
        end_ts = start_ts + t_search
    else: 
        end_ts = torch.minimum(end_ts, start_ts + t_search)

    valid_masks = (timestamps < end_ts.unsqueeze(1)) & (timestamps >= start_ts.unsqueeze(1))
    valid_masks = torch.roll(valid_masks, shifts=1, dims=1).int()

    return valid_masks

def insert_tokens(
    tokens: torch.Tensor,
    insert_masks: torch.Tensor,
    placeholder_token: int,
    pad_token: int
) -> torch.Tensor:
    """Insert before 'True' position"""

    B, S_orig = tokens.shape
    device = tokens.device

    n_insertions_per_sample = insert_masks.sum(dim=1)
    max_n_insertions = n_insertions_per_sample.max().item()

    if max_n_insertions == 0:
        return tokens

    S_new = S_orig + max_n_insertions
    new_tokens = torch.full((B, S_new), pad_token, dtype=tokens.dtype, device=device)

    shifts = torch.cumsum(insert_masks, dim=1)
    original_indices_seq = torch.arange(S_orig, device=device).expand(B, -1)
    original_dest_indices = original_indices_seq + shifts
    new_tokens.scatter_(1, original_dest_indices, tokens)
    ph_rows, ph_cols = insert_masks.nonzero(as_tuple=True)
    
    if ph_rows.numel() > 0:
        n_inserts = insert_masks[ph_rows, ph_cols]
        token_dest_indices = ph_cols + shifts[ph_rows, ph_cols]
        ph_offsets = torch.cat([torch.arange(k, 0, -1, device=device) for k in n_inserts])
        ph_dest_rows = torch.repeat_interleave(ph_rows, n_inserts)
        repeated_token_dest = torch.repeat_interleave(token_dest_indices, n_inserts)
        ph_dest_cols = repeated_token_dest - ph_offsets
        new_tokens[ph_dest_rows, ph_dest_cols] = placeholder_token

    return new_tokens


def group_argmax(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:

    unique_groups, group_indices = torch.unique(indices, return_inverse=True)
    num_groups = len(unique_groups)

    max_vals = torch.full((num_groups, len(values)), float('-inf'), device=values.device)
    pos_indices = torch.arange(len(values), device=values.device)

    max_vals[group_indices, pos_indices] = values

    indices_argmax = max_vals.argmax(dim=1)

    return indices_argmax


def group_mean(values: torch.Tensor, indices: torch.Tensor):

    values = values.flatten() 
    indices = indices.flatten()

    unique_groups, group_indices = torch.unique(indices, return_inverse=True)
    num_groups = len(unique_groups)

    group_sums = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    group_sums.scatter_add_(0, group_indices, values)

    group_counts = torch.zeros(num_groups, dtype=torch.float, device=values.device)
    group_counts.scatter_add_(0, group_indices, torch.ones_like(values, dtype=torch.float))

    group_means = group_sums / group_counts

    output = {} 
    for g, mean in zip(unique_groups, group_means): 
        output[g.item()] = mean

    return output

def compute_switch_ratio(idx_max: torch.Tensor, n_greedy_samples: int):
    n_switched = (idx_max >= n_greedy_samples).sum().item()
    switch_ratio = n_switched / n_greedy_samples
    return switch_ratio

def combine_rollout(greedy_data: torch.Tensor, greedy_data_idx: torch.Tensor, search_data: torch.Tensor, search_data_idx: torch.Tensor, pad_token_id: int):
    max_len = max(greedy_data.size(1), search_data.size(1))
    greedy_padded = torch.nn.functional.pad(
        greedy_data, (0, max_len - greedy_data.size(1)), value=pad_token_id
    )
    search_padded = torch.nn.functional.pad(
        search_data, (0, max_len - search_data.size(1)), value=pad_token_id
    )
    combined_data = torch.cat([greedy_padded, search_padded], dim=0)
    combined_data_idx = torch.cat([greedy_data_idx, search_data_idx], dim=0)
    return combined_data, combined_data_idx

