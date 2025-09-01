import torch 

from model import GAT 
from utils import HierSeq, pad_abstract_tokens
import numpy as np

from dataclasses import dataclass



# Get Batch functional 
# --------------------------------------------------------------------------------------------------------------------------

def get_batch(sequences: list, lengths: list, max_length: int, L: int, K: int):
    rand_idx = np.random.randint(0, len(sequences))
    batch = []
    curr_len = 0

    for idx in range(rand_idx, len(sequences) + rand_idx):
        idx = idx % len(sequences)
        seq = sequences[idx]
        l = lengths[idx]
        if curr_len + l > max_length:
            break
        batch.append(([seq] + [[] for _ in range(1, L)], None))
        curr_len += l

    batch_data = HierSeq.from_hierarchical_data(batch, K=K, L=L)
    return batch_data



# Relevant function for SoRL training for GAT | Essentially a combination of GRPO & SSl - abstraction trained with RL, trajectory trained with SSL
# --------------------------------------------------------------------------------------------------------------------------

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

def compute_grouped_advantage(values: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped advantage using indices to group values tensor"""
    
    unique, inverse = torch.unique(indices, return_inverse=True)
    n = len(unique)

    means = torch.zeros(n).scatter_add_(0, inverse, values) / torch.bincount(inverse).float()
    vars = torch.zeros(n).scatter_add_(0, inverse, values**2) / torch.bincount(inverse).float() - means**2
    stds = torch.sqrt(torch.clamp(vars, min=0))

    advantages = (values - means[inverse]) / (stds[inverse] + 1e-4)

    return advantages

def _compute_log_probs(ppt: torch.Tensor, repeat_batch: HierSeq) -> list:
    old_log_probs = [None]
    for l in range(1, repeat_batch.levels.max() + 1):
        level_ppt_mask = (repeat_batch.levels[1:] == l) & (repeat_batch.timestamps[1:] > 0)
        old_level_ppt = ppt[level_ppt_mask] # per-token-log-probability of un-updated model
        old_log_probs.append(old_level_ppt)
    return old_log_probs



# Rollout generation function with reference model
# --------------------------------------------------------------------------------------------------------------------------

def generate_rollout_data(old_model: GAT, ref_model: GAT,  
                          batch_data: HierSeq, n: int, 
                          temperature: float): 

    repeat_batch = repeat_hseq(batch_data, n)

    pad_abstract_tokens(repeat_batch) 

    # generate rollout 
    repeat_batch = old_model.generate(repeat_batch, parallel=True, temperature=temperature)

    # compute log_probs
    ppt = old_model(repeat_batch)
    old_log_probs = _compute_log_probs(ppt, repeat_batch) # per-level log probs

    # compute reference log_probs
    ref_ppt = ref_model(repeat_batch)
    ref_log_probs = _compute_log_probs(ref_ppt, repeat_batch) # per-level log probs

    return repeat_batch, old_log_probs, ref_log_probs



# (Tentative) GRPO loss computation functional
# --------------------------------------------------------------------------------------------------------------------------

def compute_grpo_loss(repeat_batch: HierSeq, ppt: torch.Tensor,
                      old_log_probs: list, ref_log_probs: list, 
                      epsilon: float = 0.2, beta: float = 0.1):
        
    # per-level sample_idx->reward lookup table | detach makes sense?
    reward_lookups = compute_hierarchical_rewards(ppt.detach(), repeat_batch)

    loss = torch.tensor(0.0, device=repeat_batch.tokens.device)

    for l in range(1, repeat_batch.L): 
        level_ppt_mask = (repeat_batch.levels[1:] == l) & (repeat_batch.timestamps[1:] > 0)
        new_level_ppt = ppt[level_ppt_mask] 
        old_level_ppt = old_log_probs[l] # loaded from rollout data

        if len(old_level_ppt) == 0: 
            continue 

        # Compute level l reward for each sample with abstraction at level l
        level_sample_idx = repeat_batch.sample_idx[1:][level_ppt_mask]
        sample_with_level_l = repeat_batch.indices[torch.isin(repeat_batch.indices, level_sample_idx)]

        sample_level_rewards = torch.tensor(
            [reward_lookups[l][idx.item()] for idx in sample_with_level_l],
            device=repeat_batch.tokens.device
        )
        orig_idx = torch.tensor(
            [repeat_batch.idx_map[idx.item()] for idx in sample_with_level_l],
            device=repeat_batch.tokens.device
        )

        sample_level_advantages = compute_grouped_advantage(sample_level_rewards, orig_idx)

        sample_to_advantage = {
            sample_idx.item(): adv.item() 
            for sample_idx, adv in zip(sample_with_level_l, sample_level_advantages)
        }

        advantages = torch.tensor(
            [sample_to_advantage[idx.item()] for idx in level_sample_idx],
            device=repeat_batch.tokens.device
        )

        ratio = torch.exp(new_level_ppt - old_level_ppt)

        # Compute surrogate loss
        surrogate_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages)

        # Compute KL divergence between reference & new log_probs
        log_ratio = ref_log_probs[l] - new_level_ppt
        kl = torch.exp(log_ratio) - log_ratio - 1

        # Compute per-token loss
        per_token_loss = surrogate_loss - beta * kl

        loss_abs_l = - per_token_loss.mean()

        loss += loss_abs_l

    return loss  


def compute_ssl_loss(repeat_batch: HierSeq, ppt: torch.Tensor): 

    traj_mask =(repeat_batch.levels[1:] == 0) & (repeat_batch.timestamps[1:] > 0)
    traj_ppt = ppt[traj_mask]

    ssl_loss = traj_ppt.mean()

    return ssl_loss



# Evaluation Gadget
# --------------------------------------------------------------------------------------------------------------------------
def eval_hseq(gat: GAT, batch_data: HierSeq, p_thres=4.16): 
    # remove_pad_tokens(batch_data)
    p_per_sample, critical_ts, cr_per_sample, ppt = gat(batch_data, evaluate=True, p_thres=p_thres) # per-sample avg. perplexity (different weight for each level's avg. perplexity)
    
    return p_per_sample, critical_ts, cr_per_sample, ppt


# Experiment Configuration 
# --------------------------------------------------------------------------------------------------------------------------
from model import GATConfig

@dataclass
class SORLConfig: 
    # model configuration 
    gat_config: GATConfig

    # training config
    n_generations: int = 4
    temperature: float = 1.0
    num_iterations: int = 2
    num_steps: int = 10 
    grpo_steps: int = 10 
    max_length: int = 1024 
    learning_rate: float = 1e-3

    # dataset 
    dataset_name: str = "2body_2k"
    dataset_path: str = "dataset/nbody/2body_2k.bin"

    # validation (in-domain & out-of-domain)
    id_validate_dataset_name: str = "2body_100"
    id_validate_dataset_path: str = "dataset/nbody/2body_100.bin"

    ood_validate_dataset_name: str = "3body_100"
    ood_validate_dataset_path: str = "dataset/nbody/3body_100.bin"