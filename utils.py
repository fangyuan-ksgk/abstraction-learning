from dataclasses import dataclass
from typing import List, Optional, Union
import torch, random, time
from constant import PLACE_HOLDER_STATE_TOK, PLACE_HOLDER_ACTION_TOK, MASK_TOK
from IPython.display import clear_output
from matplotlib import pyplot as plt 
import matplotlib.animation as animation
from collections import defaultdict
import numpy as np 
import struct 


# Generation Helper functions: Decide the next token level & timestamp to generate 
# --------------------------------------------------------------------------------------------------------------------------

def get_next_token_level(levels, timestamps, K, L):

    current_level = levels[-1]
    current_time = timestamps[-1]

    def make_return(level, time_offset=0):
        next_level = torch.tensor(level, dtype=levels.dtype, device=levels.device) if isinstance(level, int) else level
        next_time = current_time + time_offset
        return next_level, next_time

    if current_level == L - 1:
        return make_return(0, 1)

    mask = torch.logical_and(levels >= current_level, timestamps >= current_time - K**(current_level + 1) + 1)
    is_enough = current_time - K**(current_level + 1) + 1 >= 1 # HierSeq timestamps starts from 1
    do_plan = all(levels[mask] == current_level) & is_enough

    if do_plan: 
        return make_return(current_level + 1, 0)
    else: 
        return make_return(0, 1)


# Is it possible to create a 'loss_mask' for action / state tokens, too?
def create_loss_mask(sample_idx: torch.Tensor) -> torch.Tensor:
    sample_starts = torch.zeros_like(sample_idx, dtype=torch.bool)
    sample_starts[0] = True 
    sample_starts[1:] = sample_idx[1:] != sample_idx[:-1]
    return ~sample_starts


def make_interleave_embd(state_embd, act_embd): 
    T = state_embd.shape[0]
    interleave_embd = [act_embd[i//2] if i % 2 == 0 else state_embd[i//2] 
                for i in range(2*T-1)]
    interleave_embd = torch.stack(interleave_embd, dim=0)
    return interleave_embd

def get_unique_ordered(tensor): 
    diff = torch.cat([torch.tensor([True]), tensor[1:] != tensor[:-1]])
    unique_ordered = tensor[diff]
    return unique_ordered


# --------------------------------------------------------------------------------------------------------------------------



# Batch ver. of SeqFlat, putting multiple samples into single tensor (without batch dimension), avoids padding
# --------------------------------------------------------------------------------------------------------------------------

@dataclass
class HierSeq:

    tokens: torch.Tensor          
    levels: torch.Tensor          
    timestamps: torch.Tensor      

    sample_idx: torch.Tensor       
    batch_size: int
    K: int  
    L: int 

    idx_map: dict = None
    
    @classmethod
    def from_hierarchical_data(cls, samples_data: List[tuple], K: int, L: int,
                              sample_indices: Optional[torch.Tensor] = None):

        batch_tokens = []
        batch_levels = []
        batch_timestamps = []
        batch_sample_idx = []

        assert sample_indices is None or len(sample_indices) == len(samples_data), \
            "sample_indices must be None or have the same length as samples_data"
        
        for i, (token_seqs, timestamp_seqs) in enumerate(samples_data):

            tokens, levels, timestamps = cls._flatten_single_sample(
                token_seqs, timestamp_seqs, K, L
            )
            
            batch_tokens.append(tokens)
            batch_levels.append(levels)
            batch_timestamps.append(timestamps)
            
            if sample_indices is None: 
                sample_idx = batch_sample_idx[-1] + 1 if batch_sample_idx else 0
                batch_sample_idx += [sample_idx] * len(tokens)
            else: 
                sample_idx = sample_indices[i]
                batch_sample_idx += [sample_idx] * len(tokens)

        return cls(
            tokens=torch.cat(batch_tokens),
            levels=torch.cat(batch_levels),
            timestamps=torch.cat(batch_timestamps),
            sample_idx=torch.tensor(batch_sample_idx),
            batch_size=len(samples_data),
            K=K,
            L=L
        )
    
    def to_hierarchical_data(self):     
        samples = []
        timestamps = []
        sample_indices = torch.unique(self.sample_idx, sorted=True)
        for idx in sample_indices: 
            sample, timestamp = [], []
            for l in range(self.L): 
                sample_level_mask = (self.sample_idx == idx) & (self.levels == l)
                sample_level_tokens = self.tokens[sample_level_mask]
                sample_level_timestamps = self.timestamps[sample_level_mask]
                sample.append(sample_level_tokens.tolist())
                timestamp.append(sample_level_timestamps.tolist())
            samples.append(sample)
            timestamps.append(timestamp)
        return samples, timestamps
    

    @staticmethod
    def _flatten_single_sample(token_sequences: list, timestamp_sequences: Optional[list], K: int, L: int):
        """Flatten a single hierarchical sample by timestamp ordering."""

        if timestamp_sequences is None:
            timestamp_sequences = []
            for level in range(L):
                tokens = token_sequences[level]
                seq_len = len(tokens)
                gap = K ** level
                timestamp_sequences.append([(i + 1) * gap for i in range(seq_len)])
            
        # Collect all (timestamp, level, token) tuples
        items = []
        for level in range(L):
            tokens = token_sequences[level]
            timestamps = timestamp_sequences[level]
            
            if torch.is_tensor(tokens):
                tokens = tokens.tolist()
            
            assert len(tokens) == len(timestamps), \
                f"Level {level}: token count ({len(tokens)}) != timestamp count ({len(timestamps)})"
            
            for token, timestamp in zip(tokens, timestamps):
                items.append((timestamp, level, token))
        
        items.sort(key=lambda x: (x[0], x[1])) # Sort by timestamp, then by level
        
        sorted_tokens = [item[2] for item in items]
        sorted_levels = [item[1] for item in items]
        sorted_timestamps = [item[0] for item in items]
        
        return (
            torch.tensor(sorted_tokens, dtype=torch.long),
            torch.tensor(sorted_levels, dtype=torch.long), 
            torch.tensor(sorted_timestamps, dtype=torch.long)
        )
    
    def get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample by index."""
        
        mask = (self.sample_idx == self.indices[idx])
        
        return (
            self.tokens[mask],
            self.levels[mask], 
            self.timestamps[mask],
            mask
        )
    
    def to(self, device):
        """Move all tensors to device."""
        return HierSeq(
            tokens=self.tokens.to(device),
            levels=self.levels.to(device),
            timestamps=self.timestamps.to(device),
            sample_idx=self.sample_idx.to(device),
            batch_size=self.batch_size,
            K=self.K,
            L=self.L
        )
    
    def __len__(self):
        return self.batch_size

    def insert_tokens(self, sample_idx: int, tokens: Union[torch.Tensor, int], level: int, timestamps: Union[torch.Tensor, int]): 
        """Insert multiple tokens for a specific sample at positions determined by timestamps."""
        
        if not torch.is_tensor(tokens):
            tokens = torch.tensor([tokens])
        elif tokens.dim() == 0:
            tokens = tokens.unsqueeze(0)
            
        if not torch.is_tensor(timestamps):
            timestamps = torch.tensor([timestamps])
        elif timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
        
        if tokens.size(0) == 1 and timestamps.size(0) > 1:
            tokens = tokens.expand(timestamps.size(0))
        
        if tokens.size(0) == 0 or timestamps.size(0) == 0:
            return 
              
        assert tokens.size(0) == timestamps.size(0), "Tokens and timestamps must have same length"
        
        sample_mask = self.sample_idx == sample_idx
        if not sample_mask.any():
            raise ValueError(f"Sample {sample_idx} not found in batch")
        
        sample_positions = torch.where(sample_mask)[0]
        sample_timestamps = self.timestamps[sample_mask]
        sample_levels = self.levels[sample_mask]
        
        for token, timestamp in zip(tokens, timestamps):
            insert_after_mask = torch.logical_or(
                sample_timestamps < timestamp,
                torch.logical_and(sample_timestamps == timestamp, sample_levels < level)
            )
            
            if insert_after_mask.any():
                last_valid_pos = sample_positions[insert_after_mask][-1].item()
                insert_pos = last_valid_pos + 1
            else:
                insert_pos = sample_positions[0].item()

            suffix_mask = torch.logical_or(
                sample_timestamps > timestamp,
                torch.logical_and(sample_timestamps == timestamp, sample_levels > level)
            )
            if suffix_mask.any():
                suffix_pos = sample_positions[suffix_mask][0].item()
            else:
                suffix_pos = sample_positions[-1].item() + 1 # essentially no suffix within the sample
            
            self.tokens = torch.cat([
                self.tokens[:insert_pos],
                token.unsqueeze(0) if token.dim() == 0 else token.unsqueeze(0),
                self.tokens[suffix_pos:]
            ])
            
            self.levels = torch.cat([
                self.levels[:insert_pos],
                torch.tensor([level]),
                self.levels[suffix_pos:]
            ])
            
            self.timestamps = torch.cat([
                self.timestamps[:insert_pos],
                timestamp.unsqueeze(0) if timestamp.dim() == 0 else timestamp.unsqueeze(0),
                self.timestamps[suffix_pos:]
            ])
            
            self.sample_idx = torch.cat([
                self.sample_idx[:insert_pos],
                torch.tensor([sample_idx]),
                self.sample_idx[suffix_pos:]
            ])
            
            sample_mask = self.sample_idx == sample_idx
            sample_positions = torch.where(sample_mask)[0]
            sample_timestamps = self.timestamps[sample_mask]
            sample_levels = self.levels[sample_mask]

    @property 
    def indices(self): 
        return get_unique_ordered(self.sample_idx)

    def next_level_groups(self): 

        level_groups = defaultdict(list) 
        for b in self.indices: 
            mask = self.sample_idx == b
            l_next, t_next = get_next_token_level(
                self.levels[mask], self.timestamps[mask], self.K, self.L
            )
            level_groups[l_next.item()].append((b, mask, t_next))

        return level_groups

    def get_abstract_groups(self): 
        # Collect indices of representation used to predict abstract tokens, timestamp & level of these abstract tokens (for parallel AR generation)

        abs_groups = defaultdict(list)

        for b in self.indices: 
            mask = self.sample_idx == b
            sample_levels = self.levels[mask]
            sample_timestamps = self.timestamps[mask]

            abs_mask = sample_levels > 0 
            if abs_mask.any():
                abs_levels = sample_levels[abs_mask]
                abs_timestamps = sample_timestamps[abs_mask]

                for level in abs_levels.unique(): 
                    level_timestamps = abs_timestamps[abs_levels == level]
                    prefix_mask = torch.cat([self.levels[1:] == level, torch.tensor([False], device=self.levels.device)])
                    repr_indices = torch.where(mask & prefix_mask)[0] 
                    abs_groups[level.item()].append((torch.tensor([b]*level_timestamps.size(0)), repr_indices, level_timestamps))

        combined_groups = {} 
        for level, group in abs_groups.items(): 
            combined_groups[level] = (torch.cat([b for b, _, _ in group]), torch.cat([ind for _, ind, _ in group]), torch.cat([t for _, _, t in group]))

        return combined_groups 


    def get_pad_groups(self): 
        
        pad_groups = defaultdict(list)

        for b in self.indices: 
            mask = self.sample_idx == b
            sample_levels = self.levels[mask]
            sample_timestamps = self.timestamps[mask]
            sample_tokens = self.tokens[mask]

            pad_mask = torch.logical_and(sample_levels > 0, sample_tokens == MASK_TOK)
            if pad_mask.any():
                pad_levels = sample_levels[pad_mask]
                pad_timestamps = sample_timestamps[pad_mask]

                for level in pad_levels.unique(): 
                    level_timestamps = pad_timestamps[pad_levels == level]
                    prefix_mask = torch.cat([torch.logical_and(self.levels[1:]==level, self.tokens[1:]==MASK_TOK), torch.tensor([False], device=self.levels.device)])
                    repr_indices = torch.where(mask & prefix_mask)[0]
                    pad_groups[level.item()].append((torch.tensor([b]*level_timestamps.size(0)), repr_indices, level_timestamps))

        combined_groups = {} 
        for level, group in pad_groups.items(): 
            batch_indices = torch.cat([b for b, _, _ in group])
            mask_positions = torch.cat([ind for _, ind, _ in group])
            timestamps = torch.cat([t for _, _, t in group])
            assert batch_indices.size(0) == mask_positions.size(0) == timestamps.size(0), "mismatching batch_indices, mask_positions, timestamps number"
            combined_groups[level] = (batch_indices, mask_positions, timestamps)

        return combined_groups 

    

    def slice_prefix(self): 
        # Slice the HierSeq so that only prefix is retained
        raise NotImplementedError("Not implemented yet")


# Search Utility Functions
# --------------------------------------------------------------------------------------------------------------------------
# Caveat: using 0-th level token perplexity to infer 'prefix abstract token' timestamp is an ambiguous task. 
#       - without 'level' of the abstract token, we can only infer the 'maximal value' of its timetsamp, so 
#       - downstream utility function needs to be aware & modify it to ts  - ts % (K**l)

# Caveat: we don't add abstract token at last timestamp, it only explains non-existing future trajectories


def init_critical_ts(batch_data: HierSeq) -> torch.Tensor:

    critical_ts = torch.full((batch_data.batch_size,), -1.0, device=batch_data.tokens.device, dtype=torch.int)
    for loc_idx, glob_idx in enumerate(batch_data.indices):
        mask = batch_data.sample_idx == glob_idx
        ts, levels = batch_data.timestamps[mask], batch_data.levels[mask]
        
        for l in range(1, batch_data.L):
            K_power = batch_data.K ** l
            start = ((ts[0] - 1) // K_power + 1) * K_power  # First valid timestamp >= ts[0]
            expected_ts = torch.arange(start, ts[-1], K_power)
            existing_ts = ts[levels == l]
            missing = expected_ts[~torch.isin(expected_ts, existing_ts)]
            
            if missing.numel() > 0:
                critical_ts[loc_idx] = missing.min().int()
                break  
    
    return critical_ts


# (TBD). Add 'minimal backtrack timestamp' here to avoid 'backtrack all the way back to where previous attempt failed'
def infer_critical_ts(
    perplexity_per_token: torch.Tensor, 
    batch_data: HierSeq, 
    p_thres: float = 1.0
    ): 
    """
    Infer critical timestamps from perplexity_per_token
    """
    critical_timestamps = init_critical_ts(batch_data)

    for loc_idx, glob_idx in enumerate(batch_data.indices):
        sample_perp_mask = (batch_data.sample_idx[1:] == glob_idx) & (perplexity_per_token > p_thres) & (batch_data.levels[1:] == 0)
        if sample_perp_mask.any():
            high_perp_ts = batch_data.timestamps[1:][sample_perp_mask][0]
            critical_timestamps[loc_idx] = high_perp_ts - 1
    return critical_timestamps


def compute_cond_ratio(batch_data: HierSeq): 

    cond_ratios = []
    for glob_idx in batch_data.indices: 
        sample_mask = batch_data.sample_idx == glob_idx 
        abs_mask = (batch_data.levels > 0) & (batch_data.tokens != MASK_TOK)
        abs_count = (sample_mask & abs_mask).sum()

        et, st = batch_data.timestamps[sample_mask][[-1, 0]]
        required_abs_count = sum((et-st)//(batch_data.K**l) for l in range(1, batch_data.L))

        cond_ratio = abs_count / required_abs_count
        cond_ratios.append(cond_ratio)

    return torch.tensor(cond_ratios)

def compute_hier_seq_len(seq: list, L: int, K: int) -> int:
    """Full length with abstraction computation based on trajectory sequence alone"""
    seq_len = len(seq)
    total = seq_len
    for l in range(1, L):
        total += (seq_len - 1) // (K ** l) + 1
    return total + 1


# (TBD). Importantly, t_search is not utilized here yet, this means search is not 'progressively done'
#        rather, it's done in one go. 
def pad_abstract_tokens(batch_data: HierSeq): 
    abstract_mask = (batch_data.levels > 0)
    assert not abstract_mask.any(), " - Abstract tokens already exist, 'pad_abstract_tokens' requires no abstract tokens"

    for sample_idx in batch_data.indices: 
        sample_mask = batch_data.sample_idx == sample_idx
        sample_timestamps = batch_data.timestamps[sample_mask]
        start_ts, end_ts = sample_timestamps[0], sample_timestamps[-1]

        for l in range(1, batch_data.L): 
            abs_tok_ts = torch.arange(start_ts - 1, end_ts, batch_data.K ** l)
            batch_data.insert_tokens(sample_idx, MASK_TOK, l, abs_tok_ts[abs_tok_ts >= start_ts])

    return batch_data

def remove_pad_tokens(batch_data: HierSeq): 
    """In-place removal of PAD tokens"""
    pad_mask = torch.logical_and(batch_data.levels>0, batch_data.tokens==MASK_TOK)
    batch_data.tokens = batch_data.tokens[~pad_mask]
    batch_data.levels = batch_data.levels[~pad_mask]
    batch_data.timestamps = batch_data.timestamps[~pad_mask]
    batch_data.sample_idx = batch_data.sample_idx[~pad_mask]


def remove_abs_toks(batch_data: HierSeq, remove_ts: torch.Tensor): 
    abs_mask = (batch_data.levels > 0)
    remove_mask = torch.full(batch_data.tokens.shape, False)

    for i, sample_idx in enumerate(batch_data.indices):
        if remove_ts[i] == -1: 
            continue # no backtracking
        sample_remove_mask = (abs_mask) & (batch_data.sample_idx == sample_idx) & (batch_data.timestamps > remove_ts[i])
        remove_mask = remove_mask | sample_remove_mask
        # print(f" - remove abstract tokens to sample {sample_idx} beyond timestamp: {remove_ts[i]}")

    batch_data.tokens = batch_data.tokens[~remove_mask]
    batch_data.levels = batch_data.levels[~remove_mask]
    batch_data.timestamps = batch_data.timestamps[~remove_mask]
    batch_data.sample_idx = batch_data.sample_idx[~remove_mask]

    # sanity check (To be removed)
    level_mask = (batch_data.levels > 0)
    for i, sample_idx in enumerate(batch_data.indices): 
        sample_mask = batch_data.sample_idx == sample_idx
        if remove_ts[i] == -1: continue
        assert batch_data.timestamps[sample_mask & level_mask][-1] <= remove_ts[i], " - Remove timestamp is not correct"

def slice_prefix_before_pad(hierseq, pad_token=MASK_TOK):

    prefix_indices = []
    for b in hierseq.indices: 
        sample_mask = hierseq.sample_idx == b
        sample_levels = hierseq.levels[sample_mask]
        sample_positions = torch.where(sample_mask)[0]
        
        sample_tokens = hierseq.tokens[sample_mask]
        pad_positions = torch.where(torch.logical_and(sample_tokens == pad_token, sample_levels > 0))[0]
        
        if len(pad_positions) > 0:
            first_pad_local = pad_positions[0].item()
            prefix_indices.extend(sample_positions[:first_pad_local].tolist())
        else:
            prefix_indices.extend(sample_positions.tolist())
    
    assert len(prefix_indices)>0, "Prefix HierSeq is empty, indicating it begin with abstract tokens that are padded."
    
    prefix_indices = torch.tensor(prefix_indices)
    
    return HierSeq(
        tokens=hierseq.tokens[prefix_indices],
        levels=hierseq.levels[prefix_indices],
        timestamps=hierseq.timestamps[prefix_indices],
        sample_idx=hierseq.sample_idx[prefix_indices],
        batch_size=hierseq.batch_size,
        K=hierseq.K,
        L=hierseq.L
    )


def merge_prefix(prefix_batch, original_batch, mask_token=MASK_TOK):
    """Assume one extra abstract token generated in prefix HierSeq, use it to replace [MASK] token in original HierSeq"""
    
    n_replaced = 0
    for b in original_batch.indices: 

        orig_mask = original_batch.sample_idx == b
        prefix_mask = prefix_batch.sample_idx == b
        
        orig_positions = torch.where(orig_mask)[0]
        orig_tokens = original_batch.tokens[orig_mask]
        orig_levels = original_batch.levels[orig_mask]

        mask_locs = torch.where(torch.logical_and(orig_tokens==mask_token, orig_levels>0))[0]
        
        if len(mask_locs) == 0:
            continue  # No MASK to replace
        
        prefix_positions = torch.where(prefix_mask)[0]
        if len(prefix_positions) == 0:
            continue
            
        new_token_idx = prefix_positions[-1]
        new_token = prefix_batch.tokens[new_token_idx]
        
        first_mask_global_idx = orig_positions[mask_locs[0]]
        # print(f"Replacing {original_batch.tokens[first_mask_global_idx]} with {new_token} at {first_mask_global_idx}")

        assert original_batch.levels[first_mask_global_idx] == prefix_batch.levels[new_token_idx], \
            f"Level mismatch: expected {prefix_batch.levels[new_token_idx]}, got {original_batch.levels[first_mask_global_idx]}"
        assert original_batch.timestamps[first_mask_global_idx] == prefix_batch.timestamps[new_token_idx], \
            f"Timestamp mismatch: expected {prefix_batch.timestamps[new_token_idx]}, got {original_batch.timestamps[first_mask_global_idx]}"
        
        original_batch.tokens[first_mask_global_idx] = new_token
        n_replaced += 1

    return n_replaced


def save_tokenized_binary(sequences, tokenizer, sample_len, output_path='nbody.bin'):
    """Minimal binary save."""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', len(sequences)))  # Number of sequences
        
        for seq in sequences:
            tokens = tokenizer(seq)
            tokens = [sample_len] + tokens
            f.write(struct.pack('I', len(tokens)))  # Length of sequence
            f.write(np.array(tokens, dtype=np.int32).tobytes())  # Token data
    
    print(f"Saved {len(sequences)} sequences to {output_path}")

def load_tokenized_binary(path):
    """Minimal binary load."""
    sequences = []
    
    with open(path, 'rb') as f:
        n_sequences = struct.unpack('I', f.read(4))[0]
        
        for _ in range(n_sequences):
            sample_len = struct.unpack('I', f.read(4))[0]
            length = struct.unpack('I', f.read(4))[0]
            tokens = np.frombuffer(f.read(length * 4), dtype=np.int32)
            sequences.append((sample_len, tokens.tolist()))  # Convert to list if needed
    
    return sequences



def get_sample_level_ppl(batch_data: HierSeq, ppl_per_token: torch.Tensor, level: int): 

    per_sample_ppl, per_sample_timestamps, per_sample_max_ts = [], [], [] 

    level_mask = (batch_data.levels[1:]==level) & (batch_data.timestamps[1:] > 1)
    for sample_idx in batch_data.indices:
        sample_mask = batch_data.sample_idx[1:] == sample_idx
        per_sample_timestamps.append(batch_data.timestamps[1:][level_mask & sample_mask])
        per_sample_ppl.append(ppl_per_token[level_mask & sample_mask])
        max_abstract_ts = batch_data.timestamps[(batch_data.levels == level + 1) & (batch_data.sample_idx == sample_idx)][-1].item()
        per_sample_max_ts.append(max_abstract_ts)

    return per_sample_ppl, per_sample_timestamps, per_sample_max_ts


# (TBD). Lacks consideration of corner cases.

def get_ext_ts(batch_data: HierSeq): 
    
    ext_ts = np.array([-1 for _ in batch_data.indices])
    level_mask = (batch_data.levels > 0)
    for i, sample_idx in enumerate(batch_data.indices): 
        sample_mask = batch_data.sample_idx == sample_idx
        end_ts = batch_data.timestamps[sample_mask][-1].item()
        end_abs_ts = batch_data.timestamps[level_mask & sample_mask][-1].item()
        ext_ts[i] = end_abs_ts if end_abs_ts + (batch_data.K - 1) < end_ts else -1
    return ext_ts




# --------------------------------------------------------------------------------------------------------------------------

# HierTraj contains 'action & state'
# Assumption: HierTraj must begin with an initial state (0-th level token). 
# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class HierTraj:

    tokens: torch.Tensor          
    levels: torch.Tensor          
    timestamps: torch.Tensor      
    state_mask: torch.Tensor
    action_mask: torch.Tensor

    sample_idx: torch.Tensor       
    batch_size: int
    K: int  
    L: int 
    
    @classmethod
    def from_hierarchical_data(cls, samples_data: List[tuple], K: int, L: int):

        batch_tokens = []
        batch_levels = []
        batch_timestamps = []
        batch_state_mask = []
        batch_action_mask = []
        batch_sample_idx = []
        
        for token_seqs, timestamp_seqs in samples_data:

            tokens, levels, timestamps, state_mask, action_mask = cls._flatten_single_sample(
                token_seqs, timestamp_seqs, K, L
            )
            
            batch_tokens.append(tokens)
            batch_levels.append(levels)
            batch_timestamps.append(timestamps)
            batch_state_mask.append(state_mask)
            batch_action_mask.append(action_mask)
            sample_idx = batch_sample_idx[-1] + 1 if batch_sample_idx else 0
            batch_sample_idx += [sample_idx] * len(tokens)

        return cls(
            tokens=torch.cat(batch_tokens),
            levels=torch.cat(batch_levels),
            timestamps=torch.cat(batch_timestamps),
            state_mask=torch.cat(batch_state_mask),
            action_mask=torch.cat(batch_action_mask),
            sample_idx=torch.tensor(batch_sample_idx),
            batch_size=len(samples_data),
            K=K,
            L=L
        )
    
    @staticmethod
    def _flatten_single_sample(token_sequences, timestamp_sequences, K: int, L: int):
        """Flatten a single hierarchical trajectory by timestamp ordering."""

        timestamp_sequences = []

        for level in range(L):
            tokens = token_sequences[level]
            if torch.is_tensor(tokens):
                seq_len = tokens.size(0)
            else:
                seq_len = len(tokens)
                tokens = torch.tensor(tokens)

            if level == 0: 
                timestamp_l0 = torch.cumsum(tokens==PLACE_HOLDER_ACTION_TOK, dim=0)
                timestamp_sequences.append(timestamp_l0)
            else: 
                gap = K ** level
                timestamp_sequences.append([(i + 1) * gap for i in range(seq_len)])
            
        items = []
        for level in range(L):
            tokens = token_sequences[level]
            timestamps = timestamp_sequences[level]
            
            if torch.is_tensor(tokens):
                tokens = tokens.tolist()
            
            assert len(tokens) == len(timestamps), \
                f"Level {level}: token count ({len(tokens)}) != timestamp count ({len(timestamps)})"
            
            for token, timestamp in zip(tokens, timestamps):
                items.append((timestamp, level, token))
        
        items.sort(key=lambda x: (x[0], x[1])) # Sort by timestamp, then by level
        
        sorted_tokens = [item[2] for item in items]
        sorted_levels = [item[1] for item in items]
        sorted_timestamps = [item[0] for item in items]
        sorted_state_mask = [item[2] == PLACE_HOLDER_STATE_TOK for item in items]
        sorted_action_mask = [item[2] == PLACE_HOLDER_ACTION_TOK for item in items]
        
        return (
            torch.tensor(sorted_tokens, dtype=torch.long),
            torch.tensor(sorted_levels, dtype=torch.long), 
            torch.tensor(sorted_timestamps, dtype=torch.long),
            torch.tensor(sorted_state_mask, dtype=torch.bool),
            torch.tensor(sorted_action_mask, dtype=torch.bool)
        )

    def get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample by index."""
        
        mask = (self.sample_idx == idx)
        
        return (
            self.tokens[mask],
            self.levels[mask], 
            self.timestamps[mask],
            self.state_mask[mask],
            self.action_mask[mask],
            mask
        )
    
    def to(self, device):
        """Move all tensors to device."""
        return HierTraj(
            tokens=self.tokens.to(device),
            levels=self.levels.to(device),
            timestamps=self.timestamps.to(device),
            state_mask=self.state_mask.to(device),
            action_mask=self.action_mask.to(device),
            sample_idx=self.sample_idx.to(device),
            batch_size=self.batch_size,
            K=self.K,
            L=self.L
        )
    
    def __len__(self):
        return self.batch_size

    def insert_next_token(self, sample_idx: int, next_token: torch.Tensor, next_level: int, next_timestamp: int):
        """
        Insert next token for a specific sample at the end of that sample's sequence (in-place)
        """
        sample_positions = torch.where(self.sample_idx == sample_idx)[0]
        if len(sample_positions) == 0:
            raise ValueError(f"Sample {sample_idx} not found in batch")
        
        last_pos = sample_positions[-1].item()
        
        insert_pos = last_pos + 1
        
        if not torch.is_tensor(next_token):
            next_token = torch.tensor([next_token])
        elif next_token.dim() == 0:
            next_token = next_token.unsqueeze(0)
        
        self.tokens = torch.cat([
            self.tokens[:insert_pos], 
            next_token,
            self.tokens[insert_pos:]
        ])
        
        self.levels = torch.cat([
            self.levels[:insert_pos],
            torch.tensor([next_level]),
            self.levels[insert_pos:]
        ])
        
        self.timestamps = torch.cat([
            self.timestamps[:insert_pos],
            torch.tensor([next_timestamp]),
            self.timestamps[insert_pos:]
        ])
        
        self.sample_idx = torch.cat([
            self.sample_idx[:insert_pos],
            torch.tensor([sample_idx]),
            self.sample_idx[insert_pos:]
        ]) 

        if next_token.item() == PLACE_HOLDER_STATE_TOK:
            state_mask_item = torch.tensor([True])
        else: 
            state_mask_item = torch.tensor([False])
        self.state_mask = torch.cat([
            self.state_mask[:insert_pos],
            state_mask_item,
                self.state_mask[insert_pos:]
            ])

        if next_token.item() == PLACE_HOLDER_ACTION_TOK: 
            action_mask_item = torch.tensor([True])
        else: 
            action_mask_item = torch.tensor([False])
        self.action_mask = torch.cat([
            self.action_mask[:insert_pos],
            action_mask_item,
            self.action_mask[insert_pos:]
        ])

# --------------------------------------------------------------------------------------------------------------------------


def create_traj_loss_mask(batch_data: HierTraj): 

    sample_idx = batch_data.sample_idx
    tokens = batch_data.tokens

    batch_data.batch_size
    loss_mask_state, loss_mask_action = [], [] 
    indices = torch.unique(batch_data.sample_idx, sorted=True)
    for b in indices: 

        sample_tokens = tokens[sample_idx == b]
        ft = sample_tokens[0].item() 
        ft_state, ft_action = ft == PLACE_HOLDER_STATE_TOK, ft == PLACE_HOLDER_ACTION_TOK

        sample_state_size = sum(sample_tokens == PLACE_HOLDER_STATE_TOK).item()
        sample_action_size = sum(sample_tokens == PLACE_HOLDER_ACTION_TOK).item()

        loss_mask_state += [True] * sample_state_size if not ft_state else [False] + [True] * (sample_state_size - 1)
        loss_mask_action += [True] * sample_action_size if not ft_action else [False] + [True] * (sample_action_size - 1)

    loss_mask_state, loss_mask_action = torch.tensor(loss_mask_state), torch.tensor(loss_mask_action)

    return create_loss_mask(sample_idx)[1:], loss_mask_state, loss_mask_action



def get_next_traj_token(levels: torch.Tensor, timestamps: torch.Tensor, tokens: torch.Tensor, K: int, L: int): 

    current_level = levels[-1]
    current_time = timestamps[-1]
    next_token = None

    def make_return(level, time_offset=0, token=None):
        next_level = torch.tensor(level, dtype=levels.dtype, device=levels.device) if isinstance(level, int) else level
        next_time = current_time + time_offset
        return next_level, next_time, token

    if current_level == L - 1: 
        return make_return(0, 1, PLACE_HOLDER_ACTION_TOK)

    if current_level == 0: 
        
        mask = torch.logical_and(levels >= current_level, timestamps >= current_time - K + 1)
        flip_tokens = torch.flip(tokens[mask], dims=[0])
        is_complete = torch.cat([flip_tokens[1::2] == PLACE_HOLDER_ACTION_TOK, flip_tokens[::2] == PLACE_HOLDER_STATE_TOK]).all()
        is_enough = current_time - K + 1 >= 0
        do_plan = all(levels[mask] == current_level) & is_complete & is_enough

        if do_plan: 
            return make_return(current_level + 1, 0)
        else: 
            current_token = tokens[-1]
            if current_token == PLACE_HOLDER_ACTION_TOK: 
                return make_return(0, 0, PLACE_HOLDER_STATE_TOK)
            elif current_token == PLACE_HOLDER_STATE_TOK:
                return make_return(0, 1, PLACE_HOLDER_ACTION_TOK)
            else:
                raise ValueError(f"Invalid 0th level token: {current_token}")
    else: 
        mask = torch.logical_and(levels >= current_level, timestamps >= current_time - K**(current_level + 1) + 1)
        is_enough = current_time - K**(current_level + 1) + 1 >= 0
        do_plan = all(levels[mask] == current_level) & is_enough
        if do_plan: 
            return make_return(current_level + 1, 0)
        else: 
            return make_return(0, 1, PLACE_HOLDER_ACTION_TOK)


def _build_interleave_embd(s_embd, a_embd, ft_act): 

    n_state, n_act = s_embd.shape[0], a_embd.shape[0] if a_embd is not None else 0

    if ft_act: 
        assert n_state <= n_act <= n_state + 1, "Missing action or state token"
        traj_embd_list = [] 
        for i in range(n_act): 
            traj_embd_list.append(a_embd[i])
            if i < n_state: 
                traj_embd_list.append(s_embd[i])
    else: 
        assert n_act <= n_state <= n_act + 1, "Missing action or state token"
        traj_embd_list = [] 
        for i in range(n_state): 
            traj_embd_list.append(s_embd[i])
            if i < n_act: 
                traj_embd_list.append(a_embd[i])

    traj_embd = torch.stack(traj_embd_list, dim=0) 
    return traj_embd


# Sanity check functions
# --------------------------------------------------------------------------------------------------------------------------

def data_sanity_check(batch_data, trajectories): 
    d_len = sum([(t[0].size(0) if t[0] is not None else 0) + (t[1].size(0) if t[1] is not None else 0) for t in trajectories])
    t_len = sum(batch_data.levels == 0)
    assert d_len == t_len, f"Batch data token length mismatch with trajecotories: {d_len} != {t_len}"
    print(f"Sanity check passed: total {t_len} 0-th level tokens (state & action)")

    n_state_data = sum(batch_data.state_mask).item()
    n_state_traj = sum([(t[0].size(0) if t[0] is not None else 0) for t in trajectories])
    assert n_state_data == n_state_traj, f"State data & trajectory mismatch: {n_state_data} != {n_state_traj}"
    print(f"Sanity check passed: {n_state_data} state tokens in data, {n_state_traj} state tokens in trajectories")

    n_act_data = sum(batch_data.action_mask).item()
    n_act_traj = sum([(t[1].size(0) if t[1] is not None else 0) for t in trajectories])
    assert n_act_data == n_act_traj, f"Action data & trajectory mismatch: {n_act_data} != {n_act_traj}"
    print(f"Sanity check passed: {n_act_data} action tokens in data, {n_act_traj} action tokens in trajectories")

    tok_len = batch_data.tokens.size(0)
    assert batch_data.action_mask.size(0) == tok_len, f"Action mask length mismatch: {batch_data.action_mask.size(0)} != {tok_len}"
    assert batch_data.state_mask.size(0) == tok_len, f"State mask length mismatch: {batch_data.state_mask.size(0)} != {tok_len}"
    assert batch_data.levels.size(0) == tok_len, f"Levels length mismatch: {batch_data.levels.size(0)} != {tok_len}"
    assert batch_data.sample_idx.size(0) == tok_len, f"Sample index length mismatch: {batch_data.sample_idx.size(0)} != {tok_len}"
    assert batch_data.timestamps.size(0) == tok_len, f"Timestamps length mismatch: {batch_data.timestamps.size(0)} != {tok_len}"
    print(f"Sanity check passed: {tok_len} (action/state/abstract) tokens in data")
    return 

# Sanity check on 'HierSeq' & 'HierTraj' initialization

def test_hseq_htraj_init(K=2, L=3): 

    # (I). HierSeq init
    token_sequences = []
    

    n_tokens_highest = random.randint(2, 5)  # Highest level has fewest tokens
    
    for level in range(L-1, -1, -1):  # Go from L-1 down to 0
        if level == L-1:
            n_tokens = n_tokens_highest
        else:
            prev_n_tokens = len(token_sequences[0])  # Get count from previously added level
            expected = K * prev_n_tokens
            min_tokens = max(1, int(expected * 0.8))
            max_tokens = int(expected * 1.2)
            n_tokens = random.randint(min_tokens, max_tokens)
        
        token_sequences.insert(0, [random.randint(0, 9) for _ in range(n_tokens)])
    
    hseq = HierSeq.from_hierarchical_data([(token_sequences, None)], K, L)
    stream_print_hseq(hseq, clear=False)

    # (II). HierTraj init
    traj_sequences = []
    
    n_tokens_highest = random.randint(1, 4)
    
    for level in range(L-1, 0, -1):
        if level == L-1:
            n_tokens = n_tokens_highest
        else:
            prev_n_tokens = len(traj_sequences[0])
            expected = K * prev_n_tokens
            min_tokens = max(1, int(expected * 0.8))
            max_tokens = int(expected * 1.2)
            n_tokens = random.randint(min_tokens, max_tokens)
        
        traj_sequences.insert(0, [random.randint(0, 9) for _ in range(n_tokens)])

    if L > 1:
        expected_actions = K * len(traj_sequences[0])
        min_actions = max(2, int(expected_actions * 0.8))
        max_actions = int(expected_actions * 1.2)
        n_actions = random.randint(min_actions, max_actions)
    else:
        n_actions = random.randint(5, 12)
    
    traj_sequences.insert(0,
        [PLACE_HOLDER_STATE_TOK] + [PLACE_HOLDER_ACTION_TOK, PLACE_HOLDER_STATE_TOK] * n_actions
    )
    
    htraj = HierTraj.from_hierarchical_data([(traj_sequences, None)], K, L)
    stream_print_htraj(htraj, clear=False)


def test_gat_gen_order(gat, L=3, K=2, n_gen=20):

    sample = [[1] if level == 0 else [] for level in range(L)]
    batch_data = HierSeq.from_hierarchical_data([(sample, None)], K, L)

    for _ in range(n_gen): 
        gat.generate(batch_data)
        stream_print_hseq(batch_data)
        time.sleep(0.5)


def test_dat_gen_order(dat, env, L=3, K=2, n_gen=20): 
    from agent import _init_trajectory
    init_obs = env.reset()
    batch_data, trajectories = _init_trajectory(init_obs, K=dat.K, L=dat.L, device="cpu")

    for _ in range(n_gen): 
        dat.generate(batch_data, trajectories) 
        stream_print_htraj(batch_data, clear=True)
        time.sleep(0.5)



# Visualization functions
# --------------------------------------------------------------------------------------------------------------------------
def stream_print_hseq(batch_data: HierSeq, clear=True):
    
    if clear:
        clear_output(wait=True)
    
    tokens = batch_data.tokens.tolist() if hasattr(batch_data.tokens, 'tolist') else batch_data.tokens
    levels = batch_data.levels.tolist() if hasattr(batch_data.levels, 'tolist') else batch_data.levels
    timestamps = batch_data.timestamps.tolist() if hasattr(batch_data.timestamps, 'tolist') else batch_data.timestamps
    
    if len(tokens) == 0:
        print("No tokens to display")
        return
    
    max_level = max(levels) if levels else 0
    max_timestamp = max(timestamps) if timestamps else 0
    
    column_widths = [0] * (max_timestamp + 1)
    
    for tok, lvl, ts in zip(tokens, levels, timestamps):
        tok_str = f"[{tok}]"
        column_widths[ts] = max(column_widths[ts], len(tok_str))
    
    column_widths = [max(w, 3) + 1 for w in column_widths]  # min 3 chars + 1 space
    
    display = []
    for level in range(max_level + 1):
        row = [' ' * w for w in column_widths]  # Initialize with spaces
        display.append(row)
    
    for tok, lvl, ts in zip(tokens, levels, timestamps):
        tok_str = f"[{tok}]"
        display[lvl][ts] = tok_str.ljust(column_widths[ts])
    
    print("\n" + "="*55)
    print(f"Hierarchical Sequence K={batch_data.K} L={batch_data.L} (aligned by timestamp):")
    print("-"*55)
    
    for level in range(max_level, -1, -1):
        level_str = ''.join(display[level])
        print(f"Level {level}: {level_str}")
    
    print("="*55)
    print(f"Total tokens: {len(tokens)}, Max timestamp: {max_timestamp}")
    print()


def stream_print_htraj(batch_data: HierTraj, clear=True):
    
    if clear:
        clear_output(wait=True)
    
    tokens = batch_data.tokens.tolist() if hasattr(batch_data.tokens, 'tolist') else batch_data.tokens
    levels = batch_data.levels.tolist() if hasattr(batch_data.levels, 'tolist') else batch_data.levels
    timestamps = batch_data.timestamps.tolist() if hasattr(batch_data.timestamps, 'tolist') else batch_data.timestamps
    
    if len(tokens) == 0:
        print("No tokens to display")
        return
    
    max_level = max(levels)
    max_timestamp = max(timestamps)
    
    # First pass: determine the maximum width needed for each timestamp column
    column_widths = [0] * (max_timestamp + 1)
    
    for i, (tok, lvl, ts) in enumerate(zip(tokens, levels, timestamps)):
        if lvl == 0:
            # For level 0, we show [s] or [a]
            tok_str = '[s]' if (tok == PLACE_HOLDER_STATE_TOK or 
                               (hasattr(batch_data, 'state_mask') and batch_data.state_mask[i])) else '[a]'
        else:
            # For higher levels, show the actual token index
            tok_str = f'[{tok}]'
        
        column_widths[ts] = max(column_widths[ts], len(tok_str))
    
    # Add minimum spacing between columns (at least 1 space)
    column_widths = [max(w, 3) + 1 for w in column_widths]  # min 3 chars + 1 space
    
    # Create display grid
    display = []
    for level in range(max_level, 0, -1):  # Higher levels (2, 1)
        row = [' ' * w for w in column_widths]
        display.append(row)
    
    state_row = [' ' * w for w in column_widths]
    action_row = [' ' * w for w in column_widths]
    display.append(state_row)
    display.append(action_row)
    
    # Place tokens in the grid
    for i, (tok, lvl, ts) in enumerate(zip(tokens, levels, timestamps)):
        if lvl == 0:
            if tok == PLACE_HOLDER_STATE_TOK or (hasattr(batch_data, 'state_mask') and batch_data.state_mask[i]):
                state_row[ts] = '[s]'.ljust(column_widths[ts])
            elif tok == PLACE_HOLDER_ACTION_TOK or (hasattr(batch_data, 'action_mask') and batch_data.action_mask[i]):
                action_row[ts] = '[a]'.ljust(column_widths[ts])
        else:
            row_idx = max_level - lvl
            tok_str = f'[{tok}]'
            display[row_idx][ts] = tok_str.ljust(column_widths[ts])
    
    print("\n" + "="*55)
    print(f"Hierarchical Trajectory K={batch_data.K} L={batch_data.L} (aligned by timestamp):")
    print("-"*55)
    
    for level in range(max_level, 0, -1):
        row_idx = max_level - level
        level_str = ''.join(display[row_idx])
        print(f"Level {level}:  {level_str}")
    
    state_str = ''.join(state_row)
    action_str = ''.join(action_row)
    print(f"L0-State: {state_str}")
    print(f"L0-Action:{action_str}")
    
    print("="*55)
    print(f"Total tokens: {len(tokens)}, Max timestamp: {max_timestamp}")
    print()


def draw_gif(frames, txt="DAT agent playing snake", path="./visual/dat_snake.gif"):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # Stitch all frames from all rounds together
    all_frames = []
    frame_info = []  # Store (round_idx, frame_idx, total_frames_in_round) for each frame
    
    for round_idx, round_frames in frames.items():
        for frame_idx, frame in enumerate(round_frames):
            all_frames.append(frame)
            frame_info.append((round_idx, frame_idx, len(round_frames)))
    
    # Create animation from stitched frames
    def animate(global_frame_idx):
        ax.clear()
        ax.imshow(all_frames[global_frame_idx])
        ax.axis('off')
        
        round_idx, frame_idx, total_frames = frame_info[global_frame_idx]
        ax.set_title(f'{txt} - Round {round_idx + 1} Frame {frame_idx + 1}/{total_frames}')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(all_frames), interval=200, repeat=True)
    
    anim.save(path, writer='pillow')


def compute_dynamic_threshold(perplexities: torch.Tensor, percentile: float = 75.0) -> float:
    """Compute dynamic perplexity threshold based on population statistics"""
    return torch.quantile(perplexities[perplexities > 0], percentile / 100.0).item()

def update_critical_ts_dynamic(
    batch_data: HierSeq,
    perplexity_per_token: torch.Tensor,
    percentile: float = 75.0,
    momentum: float = 0.9,
    prev_threshold: Optional[float] = None
) -> tuple[torch.Tensor, float]:
    """Update critical timestamps with dynamic threshold using momentum"""
    current_threshold = compute_dynamic_threshold(perplexity_per_token, percentile)
    if prev_threshold is not None:
        threshold = momentum * prev_threshold + (1 - momentum) * current_threshold
    else:
        threshold = current_threshold
    
    critical_ts = infer_critical_ts(perplexity_per_token, batch_data, threshold)
    
    # Re-vitalize sequences if threshold dropped significantly
    if prev_threshold and threshold < prev_threshold * 0.8:
        mask = compute_cond_ratio(batch_data) > 0.9
        critical_ts[mask] = init_critical_ts(batch_data)[mask]
    
    return critical_ts, threshold
