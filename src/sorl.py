from src.regat import GAT
from src.utils import infer_level, infer_timestamp, infer_rythmic_insertion_mask, insert_tokens, infer_spike_insertion_mask, infer_valid_masks, group_argmax
from copy import deepcopy
import torch
from typing import Optional, Union 
from dataclasses import dataclass

@dataclass 
class SORLConfig: 
    l: int # level of abstraction to search & learn
    n: int # number of candidates to rollout 
    temperature: float 
    steps: int  # steps for chunk-wise denoise
    abstract_budget: int # max number of spiky abstraction allowed
    use_rhythmic_placeholders: bool = True # whether to use rhythmic placeholders
    use_spike_placeholders: bool = True # whether to use spike placeholders


# Placeholder addition function (for parallel search & training)
# -----------------------------------------------------------------------------------------------------
def add_rhythmic_placeholders(model: GAT, idx: torch.Tensor, l: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None): 
    
    tokens = deepcopy(idx)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    insert_masks = infer_rythmic_insertion_mask(levels, timestamps, model.K, l)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks.int(), model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens
    
def add_spike_placeholders(model: GAT, data: torch.Tensor, l: int, abstract_budget: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None): 

    tokens = deepcopy(data)
    idx = tokens[:, :-1].contiguous()
    target = tokens[:, 1:].contiguous()
    with torch.no_grad():
        ppt = model(idx, target)

    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    insert_masks = infer_spike_insertion_mask(levels, ppt, l, abstract_budget)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks, model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens

def pad_abstract_tokens(tokens: torch.Tensor, 
                        gat: GAT,
                        l: int,
                        t_search: Optional[Union[int, torch.Tensor]] = None,
                        start_ts: Optional[torch.Tensor] = None, 
                        use_spike_placeholders: bool = False,
                        abstract_budget: Optional[int] = None,
                        use_rhythmic_placeholders: bool = False):

    if use_spike_placeholders:
        assert abstract_budget is not None, "abstract_budgets must be provided for spike placeholders"

    if use_spike_placeholders:
        batch_data = add_spike_placeholders(gat, tokens, l, abstract_budget, t_search, start_ts)

    if use_rhythmic_placeholders:
        batch_data = add_rhythmic_placeholders(gat, tokens, l, t_search, start_ts)

    return batch_data

def repad_abstract_tokens(tokens: torch.Tensor, model: GAT, l: int, start_ts: torch.Tensor): 
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    repad_mask = (timestamps >= start_ts.unsqueeze(1)) & (levels == l)
    tokens[repad_mask] = model.level_mask_tokens[l]
    return tokens

def prep_denoise(tokens: torch.Tensor, model: GAT):
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    denoise_mask = torch.isin(tokens, model.level_mask_tokens)
    denoise_levels = levels[denoise_mask]
    return denoise_mask, denoise_levels

def chunk_denoise(data: torch.Tensor, model: GAT, l: int, steps: int, 
                   max_t_search: Optional[int] = None, temperature: float = 0.0):

    tokens = deepcopy(data)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    max_ts = timestamps.max(dim=1).values
    if max_t_search is not None:
        max_ts = torch.minimum(max_ts, torch.tensor(max_t_search))

    search_ts = torch.ceil(max_ts / steps)

    for step in range(steps): 

        start_ts = (search_ts * step).int()

        # What we actually need here is "repad" -- not new tricks required, just replace abstract tokens with "mask" from certain timestamp onwards
        tokens = repad_abstract_tokens(tokens, model, l, start_ts)

        denoise_mask, denoise_levels = prep_denoise(tokens, model)

        if denoise_levels.numel() > 0:
            with torch.no_grad():  
                tokens = model.denoise(tokens, denoise_mask, denoise_levels, temperature=temperature)

        if (start_ts >= max_ts).all(): 
            break 

    return tokens

def generate_rollout_data(data: torch.Tensor, model: GAT, l: int, 
                          n: int, temperature: float,
                          steps: int, max_t_search: Optional[int] = None,
                          t_search: Optional[int] = None, start_ts: Optional[int] = None, use_spike_placeholders: bool = True,
                          abstract_budget: Optional[int] = 5, use_rhythmic_placeholders: bool = True):
    # Generate Rollout Data 
    data_idx = torch.arange(data.shape[0])
    # (1). Repeat Data | Need to record indices of repeated data for later selection part 
    repeat_data = data.repeat_interleave(n, dim=0)
    repeat_data_idx = data_idx.repeat_interleave(n, dim=0)
    # (2). Pad abstract placeholders
    repeat_data = pad_abstract_tokens(repeat_data, model, l, t_search=t_search, start_ts=start_ts, use_spike_placeholders=use_spike_placeholders, abstract_budget=abstract_budget, use_rhythmic_placeholders=use_rhythmic_placeholders)
    # (3). Denoise 
    repeat_data = chunk_denoise(repeat_data, model, l, steps=steps, max_t_search=max_t_search, temperature=temperature)
    return repeat_data, repeat_data_idx

def sorl_search(data: torch.Tensor, model: GAT, config: SORLConfig): 

    # greedy-involved rollout
    greedy_data, greedy_data_idx = generate_rollout_data(data, model, l=config.l, n=config.n, temperature=0., steps=config.steps)
    search_data, search_data_idx = generate_rollout_data(data, model, l=config.l, n=config.n, temperature=config.temperature, steps=config.steps)

    combined_data = torch.cat([greedy_data, search_data], dim=0)
    combined_data_idx = torch.cat([greedy_data_idx, search_data_idx], dim=0)

    with torch.no_grad():
        ppt = model(idx = combined_data[:, :-1].contiguous(), 
                    target = combined_data[:, 1:].contiguous())

    # select best for each sample idx
    idx_max = group_argmax(ppt.mean(axis=1), combined_data_idx)
    best_data = combined_data[idx_max]

    return best_data