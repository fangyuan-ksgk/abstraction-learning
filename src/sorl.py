from src.regat import GAT
from src.utils import infer_level, infer_timestamp, infer_rythmic_insertion_mask, insert_tokens, infer_spike_insertion_mask
from copy import deepcopy
import torch
from typing import Optional


def add_rhythmic_placeholders(model: GAT, idx: torch.Tensor, l: int, t_search: Optional[int] = None): 
    
    tokens = deepcopy(idx)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    insert_masks = infer_rythmic_insertion_mask(levels, timestamps, model.K, l)

    if t_search is not None: 
        valid_masks = torch.roll(timestamps < t_search, shifts=1, dims=1).int()
        insert_masks *= valid_masks
        
    tokens = insert_tokens(tokens, insert_masks.int(), model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens

    
def add_spike_placeholders(model: GAT, data: torch.Tensor, l: int, abstract_budget: int, t_search: Optional[int] = None): 

    tokens = deepcopy(data)
    idx = tokens[:, :-1].contiguous()
    target = tokens[:, 1:].contiguous()
    with torch.no_grad():
        ppt = model(idx, target)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    insert_masks = infer_spike_insertion_mask(levels, ppt, l, abstract_budget)

    if t_search is not None: 
        valid_masks = torch.roll(timestamps < t_search, shifts=1, dims=1).int()
        insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks, model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens