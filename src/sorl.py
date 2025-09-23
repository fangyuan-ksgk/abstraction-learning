from src.gat import GAT
from src.utils import infer_level, infer_timestamp, infer_rythmic_insertion_mask, insert_tokens, infer_spike_insertion_mask, infer_valid_masks, group_argmax, group_mean
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
    max_t_search: Optional[int] = None # max number of tokens to search for abstraction
    start_ts: Optional[torch.Tensor] = None # start timestamp for adding placeholders | one for each sample
    end_ts: Optional[torch.Tensor] = None # end timestamp for adding placeholders | one for each sample
    abstract_budget: int = 5 # max number of spiky abstraction allowed
    use_rhythmic_placeholders: bool = True # whether to use rhythmic placeholders
    use_spike_placeholders: bool = True # whether to use spike placeholders
    causal_rollout: bool = False # whether to use causal rollout


# Placeholder addition function (for parallel search & training)
# -----------------------------------------------------------------------------------------------------
def add_rhythmic_placeholders(model: GAT, idx: torch.Tensor, l: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None, end_ts: Optional[torch.Tensor] = None): 
    
    tokens = deepcopy(idx)
    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    insert_masks = infer_rythmic_insertion_mask(levels, timestamps, model.K, l)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search, end_ts)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks.int(), model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens
    
def add_spike_placeholders(model: GAT, data: torch.Tensor, l: int, abstract_budget: int, t_search: Optional[Union[int, torch.Tensor]] = None, start_ts: Optional[torch.Tensor] = None, end_ts: Optional[torch.Tensor] = None): 

    tokens = deepcopy(data)
    idx = tokens[:, :-1].contiguous()
    target = tokens[:, 1:].contiguous()
    with torch.no_grad():
        ppt = model(idx, target)

    levels = infer_level(tokens, model.vocab_sizes, model.level_mask_tokens[0])
    timestamps = infer_timestamp(levels, model.K, l)
    insert_masks = infer_spike_insertion_mask(levels, ppt, l, abstract_budget)

    valid_masks = infer_valid_masks(timestamps, start_ts, t_search, end_ts)
    insert_masks *= valid_masks

    tokens = insert_tokens(tokens, insert_masks, model.level_mask_tokens[l], model.level_mask_tokens[0])

    return tokens

def pad_abstract_tokens(tokens: torch.Tensor, 
                        gat: GAT,
                        l: int,
                        t_search: Optional[Union[int, torch.Tensor]] = None,
                        start_ts: Optional[torch.Tensor] = None, 
                        end_ts: Optional[torch.Tensor] = None,
                        use_spike_placeholders: bool = False,
                        abstract_budget: Optional[int] = None,
                        use_rhythmic_placeholders: bool = False):

    if use_spike_placeholders:
        assert abstract_budget is not None, "abstract_budgets must be provided for spike placeholders"

    if use_spike_placeholders:
        batch_data = add_spike_placeholders(gat, tokens, l, abstract_budget, t_search, start_ts, end_ts)

    if use_rhythmic_placeholders:
        batch_data = add_rhythmic_placeholders(gat, tokens, l, t_search, start_ts, end_ts)

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

def heuristic_rollout(data: torch.Tensor, model: GAT, l: int, 
                          n: int, temperature: float,
                          steps: int, max_t_search: Optional[int] = None,
                          start_ts: Optional[torch.Tensor] = None, end_ts: Optional[torch.Tensor] = None, 
                          use_spike_placeholders: bool = True, abstract_budget: Optional[int] = 5, use_rhythmic_placeholders: bool = True):
    """Heuristic-based decision on when to generate abstraction"""

    data_idx = torch.arange(data.shape[0])

    repeat_data = data.repeat_interleave(n, dim=0)
    repeat_data_idx = data_idx.repeat_interleave(n, dim=0)

    repeat_data = pad_abstract_tokens(repeat_data, model, l, t_search=max_t_search, start_ts=start_ts, end_ts=end_ts, use_spike_placeholders=use_spike_placeholders, abstract_budget=abstract_budget, use_rhythmic_placeholders=use_rhythmic_placeholders)

    repeat_data = chunk_denoise(repeat_data, model, l, steps=steps, max_t_search=max_t_search, temperature=temperature)

    return repeat_data, repeat_data_idx

# Most beautiful way of doing rollout -- no heuristic, no manual placeholders, just model-based decision to maximize 'reward'
def causal_generate(data: torch.Tensor, model: GAT, temperature: float): 
    """Model-based decision on when to generate abstraction"""

    pad_token_id = model.level_mask_tokens[0]   
    kv_cache, levels = None, None
    progress_idx = torch.zeros(data.size(0), dtype=torch.long)
    current_idx = data[torch.arange(data.size(0)), progress_idx].contiguous()

    new_data = current_idx.unsqueeze(1)

    while torch.any(progress_idx < data.size(1) - 1): 

        not_finished_mask = progress_idx < data.size(1) - 1
        next_idx, kv_cache, levels = model.generate(current_idx.unsqueeze(1), temperature=temperature, kv_cache=kv_cache, levels=levels)
        next_level = infer_level(next_idx, model.vocab_sizes, model.level_mask_tokens[0])
        traj_mask = (next_level == 0)
        effective_traj_mask = traj_mask & not_finished_mask
        
        current_idx = next_idx
        
        if torch.any(effective_traj_mask):
            safe_indices = torch.clamp(progress_idx + 1, max=data.size(1) - 1)
            gt_tokens = data[torch.arange(data.size(0)), safe_indices]
            current_idx[effective_traj_mask] = gt_tokens[effective_traj_mask]
        
        progress_idx += effective_traj_mask.long()
        next_column_to_append = torch.full_like(current_idx, fill_value=pad_token_id)
        next_column_to_append[not_finished_mask] = current_idx[not_finished_mask]
        new_data = torch.cat([new_data, next_column_to_append.unsqueeze(1)], dim=1)

    return new_data


def causal_rollout(data: torch.Tensor, model: GAT, temperature: float, n: int):
    """The simplificty of this procedure speaks for itself"""
    repeat_data = data.repeat_interleave(n, dim=0)
    repeat_data_idx = torch.arange(data.shape[0]).repeat_interleave(n, dim=0)
    repeat_data = causal_generate(repeat_data, model, temperature)
    return repeat_data, repeat_data_idx


def sorl_search(data: torch.Tensor, model: GAT, config: SORLConfig): 

    # greedy-involved rollout
    assert config.n > 1, "n must be greater than 1"
    if config.causal_rollout:
        greedy_data, greedy_data_idx = causal_rollout(data, model, temperature=0., n=1)
        search_data, search_data_idx = causal_rollout(data, model, temperature=config.temperature, n=config.n-1)
    else:
        greedy_data, greedy_data_idx = heuristic_rollout(data, model, l=config.l, n=1, temperature=0., steps=config.steps, max_t_search=config.max_t_search, start_ts=config.start_ts, end_ts=config.end_ts, use_spike_placeholders=config.use_spike_placeholders, abstract_budget=config.abstract_budget, use_rhythmic_placeholders=config.use_rhythmic_placeholders)
        search_data, search_data_idx = heuristic_rollout(data, model, l=config.l, n=config.n-1, temperature=config.temperature, steps=config.steps, max_t_search=config.max_t_search, start_ts=config.start_ts, end_ts=config.end_ts, use_spike_placeholders=config.use_spike_placeholders, abstract_budget=config.abstract_budget, use_rhythmic_placeholders=config.use_rhythmic_placeholders)

    combined_data = torch.cat([greedy_data, search_data], dim=0)
    combined_data_idx = torch.cat([greedy_data_idx, search_data_idx], dim=0)

    with torch.no_grad():
        ppt = model(idx = combined_data[:, :-1].contiguous(), 
                    target = combined_data[:, 1:].contiguous())

    # select best for each sample idx
    idx_max = group_argmax(ppt.mean(axis=1), combined_data_idx)
    best_data = combined_data[idx_max]

    return best_data

# loss computation (level-0 is trajectory loss, level >= 1 is abstract loss)
# ------------------------------------------------------------------------------------------------
def compute_loss(data: torch.Tensor, model: GAT, ppt: torch.Tensor): 
    levels = infer_level(data, model.vocab_sizes, model.level_mask_tokens[0])
    level_loss = {l: torch.tensor(0.) for l in range(model.L)}
    level_loss.update(group_mean(ppt, levels[:, 1:]))
    return level_loss

# Sub-optimal way of evaluating search improvement || We'd like to have "evaluate" function that properly does token-by-token generation
# ---------------------------------------------------------------------------------------------------------------------------------------

def eval_gat(data: torch.Tensor, model: GAT, n: int, config: SORLConfig): 
    """Causal rollout do not assume full trajectory to add abstraction, it instead perform causal generation, suited for evaluation"""

    with torch.no_grad():        
        assert n > 1, "n must be greater than 1"
        greedy_data, _ = causal_rollout(data, model, temperature=0., n=1, max_t_search=config.max_t_search)
        search_data, _ = causal_rollout(data, model, temperature=100., n=n-1, max_t_search=config.max_t_search)

        greedy_ppt = model(greedy_data[:, :-1].contiguous(), greedy_data[:, 1:].contiguous())
        search_ppt = model(search_data[:, :-1].contiguous(), search_data[:, 1:].contiguous())

    greedy_ppl = compute_loss(greedy_data, model, greedy_ppt)[0]
    search_ppl = compute_loss(search_data, model, search_ppt)[0]

    improve_ppl_percentage = (search_ppl - greedy_ppl) / search_ppl # percentage of improvement in ppl

    return greedy_ppl, improve_ppl_percentage.mean() * 100