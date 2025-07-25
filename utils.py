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

    B = levels.shape[0]
    next_levels = torch.zeros(B, dtype=torch.long, device=levels.device)
    next_timestamps = torch.zeros(B, dtype=torch.long, device=timestamps.device)
    
    for b in range(B):
        batch_levels = levels[b]
        batch_timestamps = timestamps[b]
        
        max_timestamp = batch_timestamps.max().item()
        
        current_level = batch_levels[batch_timestamps == max_timestamp][0].item() 

        if current_level == L - 1:
            next_levels[b] = 0
            next_timestamps[b] = max_timestamp + 1
            continue
        
        consecutive_count = 0
        
        for i in range(K):
            target_timestamp = max_timestamp - K + 1 + i
            mask = (batch_timestamps == target_timestamp) & (batch_levels == current_level)
            if mask.any():
                consecutive_count += 1
            else:
                break
        
        if consecutive_count == K:
            next_levels[b] = current_level + 1
            next_timestamps[b] = max_timestamp
        else:
            next_levels[b] = current_level
            next_timestamps[b] = max_timestamp + 1
    
    return next_levels, next_timestamps

# --------------------------------------------------------------------------------------------------------------------------