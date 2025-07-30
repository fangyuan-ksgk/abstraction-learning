from dataclasses import dataclass
from typing import List, Optional, Union
import torch


# Generation Helper functions: Decide the next token level & timestamp to generate 
# --------------------------------------------------------------------------------------------------------------------------

def get_next_token_level(levels, timestamps, K, L):

    current_level = levels[-1]
    current_time = timestamps[-1]

    if current_level == L - 1:
        next_level = torch.tensor(0, dtype=levels.dtype, device=levels.device)
        next_timestamp = current_time + 1
        return next_level, next_timestamp

    mask = torch.logical_and(levels >= current_level, timestamps >= current_time - K**(current_level + 1) + 1)
    do_plan = all(levels[mask] == current_level)
    if do_plan: 
        next_level = current_level + 1
        next_timestamp = current_time
    else: 
        next_level = torch.tensor(0, dtype=levels.dtype, device=levels.device)
        next_timestamp = current_time + 1

    return next_level, next_timestamp


def create_loss_mask(sample_idx: torch.Tensor) -> torch.Tensor:
    sample_starts = torch.zeros_like(sample_idx, dtype=torch.bool)
    sample_starts[0] = True 
    sample_starts[1:] = sample_idx[1:] != sample_idx[:-1]
    return ~sample_starts

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
        
        mask = (self.sample_idx == idx)
        
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

# --------------------------------------------------------------------------------------------------------------------------

# HierSeq only contains 'state', while HierTraj contains 'action & state'
# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class HierTraj(HierSeq):

    @staticmethod
    def _flatten_single_sample(token_sequences, timestamp_sequences, K: int, L: int):
        """Flatten a single hierarchical sample by timestamp ordering."""

        if timestamp_sequences is None:
            timestamp_sequences = []
            for level in range(L):
                tokens = token_sequences[level]
                if torch.is_tensor(tokens):
                    seq_len = tokens.size(0)
                    tokens = tokens.tolist()
                else:
                    seq_len = len(tokens)
                
                if level == 0: 
                    timestamp_sequences.append([i//2 for i in range(seq_len + 1)][1:]) # [0, 1, 1, 2, 2, 3, 3, ...]
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
        
        return (
            torch.tensor(sorted_tokens, dtype=torch.long),
            torch.tensor(sorted_levels, dtype=torch.long), 
            torch.tensor(sorted_timestamps, dtype=torch.long)
        )

# --------------------------------------------------------------------------------------------------------------------------



