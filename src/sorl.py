from src.regat import GAT
from src.utils import infer_level, infer_timestamp, infer_rythmic_insertion_mask, insert_tokens, infer_spike_insertion_mask, infer_valid_masks
from copy import deepcopy
import torch
from typing import Optional, Union 

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


def prep_denoise(tokens: torch.Tensor, model: GAT, l: int): 
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    denoise_mask = (levels == model.level_mask_tokens[l])
    denoise_levels = levels[denoise_mask]
    denoise_mask = torch.roll(denoise_mask, shifts=-1, dims=1).bool()
    return denoise_mask, denoise_levels


def chunk_denoise(data: torch.Tensor, model: GAT, l: int, steps: int, 
                   max_t_search: Optional[int] = None,
                   use_spike_placeholders: bool = False,
                   abstract_budget: Optional[int] = None,
                   use_rhythmic_placeholders: bool = False):

    tokens = deepcopy(data)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    max_ts = timestamps.max(dim=1).values
    if max_t_search is not None:
        max_ts = torch.minimum(max_ts, torch.tensor(max_t_search))

    search_ts = torch.ceil(max_ts / steps)

    for step in range(steps): 

        start_ts = (search_ts * step).int()

        tokens = pad_abstract_tokens(tokens, model, l, None, start_ts, 
                            use_spike_placeholders, abstract_budget, use_rhythmic_placeholders)

        denoise_mask, denoise_levels = prep_denoise(tokens, model, l)

        if denoise_levels.numel() > 0:
            with torch.no_grad():  
                tokens = model.denoise(tokens, denoise_mask, denoise_levels, temperature=0.0)

        if (start_ts >= max_ts).all(): 
            break 

    return tokens