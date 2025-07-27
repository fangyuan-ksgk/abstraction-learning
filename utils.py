from dataclasses import dataclass
from typing import List, Optional, Union
import torch



# Hiearchical sequence flattener 
# list of seq per abstraction level & timestamp --> causal ordered sequence & level index
# for causal attention (beyond pure temporal causality, suitable for GAT)
# Remark 1. The token/timestamp seqs can totally be sparse (we could have token at t=1, t=9 instead of t=1,2,3,4,5,6,7,8,9)
#           This directly supports the memory compression scheme. 
# Remark 2. For generation purpose, we are missing 'next token timestamp' & 'next token level' informations.
#           This informs the model what to predict next (which lm-head to use) 
# --------------------------------------------------------------------------------------------------------------------------


class SeqFlat:
    
    def __init__(self, K: int, L: int):
        self.K = K
        self.L = L
    
    def __call__(self, token_sequences, timestamp_sequences=None):
       
        if timestamp_sequences is None:
            timestamp_sequences = self._generate_default_timestamps(token_sequences)
        
        assert len(token_sequences) == self.L, f"Expected {self.L} token sequences, got {len(token_sequences)}"
        assert len(timestamp_sequences) == self.L, f"Expected {self.L} timestamp sequences, got {len(timestamp_sequences)}"
        
        first_tokens = token_sequences[0]
        is_batched = torch.is_tensor(first_tokens) and first_tokens.dim() > 1
        
        if is_batched:
            batch_size = first_tokens.size(0)
            batch_tokens = []
            batch_levels = []
            batch_timestamps = []
            
            for b in range(batch_size):
                batch_token_seqs = []
                for level in range(self.L):
                    if torch.is_tensor(token_sequences[level]):
                        batch_token_seqs.append(token_sequences[level][b].tolist())
                    else:
                        batch_token_seqs.append(token_sequences[level])  # Assume same for all batches if not tensor
                
                tokens, levels = self._process_single_sample(batch_token_seqs, timestamp_sequences)
                batch_tokens.append(tokens)
                batch_levels.append(levels)
                batch_timestamps.append(timestamp_sequences[level])
                
            return torch.stack(batch_tokens), torch.stack(batch_levels), torch.stack(batch_timestamps)
        else:
            token_seqs_list = []
            for level in range(self.L):
                tokens = token_sequences[level]
                if torch.is_tensor(tokens):
                    tokens = tokens.tolist()
                token_seqs_list.append(tokens)
            
            idx, levels, timestamps = self._process_single_sample(token_seqs_list, timestamp_sequences)
            # return idx, levels, timestamps
            return idx.unsqueeze(0), levels.unsqueeze(0), timestamps.unsqueeze(0)
    
    def _process_single_sample(self, token_sequences, timestamp_sequences):
        # this function should also return 'next token timestamp & level' 
        items = []
        for level in range(self.L):
            tokens = token_sequences[level]
            timestamps = timestamp_sequences[level]
            
            assert len(tokens) == len(timestamps), \
                f"Level {level}: token count ({len(tokens)}) != timestamp count ({len(timestamps)})"
            
            for token, timestamp in zip(tokens, timestamps):
                items.append((timestamp, level, token))
        
        items.sort(key=lambda x: (x[0], x[1]))
        
        sorted_tokens = [item[2] for item in items]
        sorted_levels = [item[1] for item in items]
        sorted_timestamps = [item[0] for item in items]
        
        return torch.tensor(sorted_tokens, dtype=torch.long), torch.tensor(sorted_levels, dtype=torch.long), torch.tensor(sorted_timestamps, dtype=torch.long)
    
    def _get_next_token_info(self, sorted_timestamps, sorted_levels): 
        curr_t = sorted_timestamps[-1]
        curr_l = sorted_levels[-1]
        if curr_t % (self.K**(curr_l + 1)) == 0: # if timestamp begins with 1, otherwise we need (curr_t + 1)
            next_t = curr_t
            next_l = curr_l + 1
        else: 
            next_t = curr_t + 1
            next_l = 0


    def _generate_default_timestamps(self, token_sequences):
        timestamp_sequences = []
        for level in range(self.L):
            tokens = token_sequences[level]
            if torch.is_tensor(tokens):
                if tokens.dim() > 1:  # Batched
                    seq_len = tokens.size(-1)  
                else:
                    seq_len = tokens.size(0)
            else: 
                seq_len = len(tokens)
                
            gap = self.K ** level
            timestamp_sequences.append([(i + 1) * gap for i in range(seq_len)])
        
        return timestamp_sequences

# --------------------------------------------------------------------------------------------------------------------------


# Generation Helper functions: Decide the next token level & timestamp to generate 
# --------------------------------------------------------------------------------------------------------------------------

def get_next_token_level(levels, timestamps, K, L):
    
    max_timestamp = timestamps.max().item()
    current_level = levels[timestamps == max_timestamp][0].item()

    if current_level == L - 1:
        next_level[0] = 0
        next_timestamp[0] = max_timestamp + 1
        return next_level, next_timestamp
    
    consecutive_count = 0
    for i in range(K):
        target_timestamp = max_timestamp - K + 1 + i
        mask = (timestamps == target_timestamp) & (levels == current_level)
        if mask.any():
            consecutive_count += 1
        else:
            break
    
    if consecutive_count == K:
        next_level = current_level + 1
        next_timestamp = max_timestamp
    else:
        next_level = current_level
        next_timestamp = max_timestamp + 1

    return next_level, next_timestamp


def update_idx_seq(idx_seq, t_seq, next_token, next_level, next_timestamp):
        
    assert len(idx_seq) > next_level, "idx_seq is not long enough, current level is {}, but idx_seq has only {} levels".format(next_level, len(idx_seq))
    
    if isinstance(idx_seq[next_level], list): 
        idx_seq[next_level].append(next_token.item() if isinstance(next_token, torch.Tensor) else next_token)
    else: 
        idx_seq[next_level] = torch.cat([idx_seq[next_level], next_token if isinstance(next_token, torch.Tensor) else torch.tensor([next_token])])

    if t_seq is not None: 
        if isinstance(t_seq[next_level], list): 
            t_seq[next_level].append(next_timestamp)
        else: 
            t_seq[next_level] = torch.cat([t_seq[next_level], torch.tensor([next_timestamp])])

    return idx_seq, t_seq

# --------------------------------------------------------------------------------------------------------------------------



# Batch ver. of SeqFlat, putting multiple samples into single tensor (without batch dimension), avoids padding
# --------------------------------------------------------------------------------------------------------------------------

@dataclass
class BatchedHierarchicalData:

    tokens: torch.Tensor          
    levels: torch.Tensor          
    timestamps: torch.Tensor      

    sample_idx: torch.Tensor       
    batch_size: int
    K: int  
    L: int 
    
    @classmethod
    def from_hierarchical_data(cls, samples_data: List[tuple], K: int, L: int):

        batch_tokens = []
        batch_levels = []
        batch_timestamps = []
        batch_sample_idx = []
        
        for token_seqs, timestamp_seqs in samples_data:

            tokens, levels, timestamps = cls._flatten_single_sample(
                token_seqs, timestamp_seqs, K, L
            )
            
            batch_tokens.append(tokens)
            batch_levels.append(levels)
            batch_timestamps.append(timestamps)
            sample_idx = batch_sample_idx[-1] + 1 if batch_sample_idx else 0
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
    
    @staticmethod
    def _flatten_single_sample(token_sequences, timestamp_sequences, K: int, L: int):
        """Flatten a single hierarchical sample by timestamp ordering."""
        # Generate default timestamps if None
        if timestamp_sequences is None:
            timestamp_sequences = []
            for level in range(L):
                tokens = token_sequences[level]
                if torch.is_tensor(tokens):
                    seq_len = tokens.size(0)
                    tokens = tokens.tolist()
                else:
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
        
        # Sort by timestamp, then by level
        items.sort(key=lambda x: (x[0], x[1]))
        
        # Extract sorted sequences
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
        
        mask = (self.sample_idx == idx)
        
        return (
            self.tokens[mask],
            self.levels[mask], 
            self.timestamps[mask]
        )
    
    def to(self, device):
        """Move all tensors to device."""
        return BatchedHierarchicalData(
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
    
    def get_next_token_info(self, sample_idx: int):
        """Get next token timestamp and level for a given sample."""
        tokens, levels, timestamps = self.get_sample(sample_idx)
        
        if len(timestamps) == 0:
            return 1, 0  # Start with timestamp 1, level 0
            
        curr_t = timestamps[-1].item()
        curr_l = levels[-1].item()
        
        # Check if we should generate next level token
        if curr_t % (self.K ** (curr_l + 1)) == 0:
            next_t = curr_t
            next_l = curr_l + 1
        else:
            next_t = curr_t + 1
            next_l = 0
            
        return next_t, next_l

# --------------------------------------------------------------------------------------------------------------------------