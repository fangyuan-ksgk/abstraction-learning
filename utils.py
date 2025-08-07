from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from constant import PLACE_HOLDER_STATE_TOK, PLACE_HOLDER_ACTION_TOK
from IPython.display import clear_output


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
    is_enough = timestamps[mask][0] == (current_time - K**(current_level + 1) + 1)
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
    
        if timestamp_sequences is None:
            timestamp_sequences = []

            tokens = token_sequences[0]
            if isinstance(tokens, list): tokens = torch.tensor(tokens)
            l0_timestamp = torch.cumsum(tokens==PLACE_HOLDER_ACTION_TOK, dim=0)
            timestamp_sequences.append(l0_timestamp)

            for level in range(1, L):
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

    def insert_next_token(self, sample_idx: int, next_token: Union[torch.Tensor, int], next_level: int, next_timestamp: int):
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
    for b in range(batch_data.batch_size): 

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


def stream_print_hierarchy(batch_data, clear=True):
    
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
    
    display = []
    for level in range(max_level, 0, -1):  # Higher levels (2, 1)
        row = ['     '] * (max_timestamp + 1)  # 5 spaces to match "[s] " width
        display.append(row)
    
    state_row = ['     '] * (max_timestamp + 1)
    action_row = ['     '] * (max_timestamp + 1)
    display.append(state_row)
    display.append(action_row)
    
    for i, (tok, lvl, ts) in enumerate(zip(tokens, levels, timestamps)):
        if lvl == 0:
            if tok == PLACE_HOLDER_STATE_TOK or (hasattr(batch_data, 'state_mask') and batch_data.state_mask[i]):
                state_row[ts] = '[s]  '
            elif tok == PLACE_HOLDER_ACTION_TOK or (hasattr(batch_data, 'action_mask') and batch_data.action_mask[i]):
                action_row[ts] = '[a]  '
        else:
            row_idx = max_level - lvl
            display[row_idx][ts] = '[t]  '
    
    print("\n" + "="*55)
    print("Hierarchical Token Stream (aligned by timestamp):")
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


