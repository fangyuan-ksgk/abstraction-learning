import torch



# Hiearchical sequence flattener 
# list of seq per abstraction level & timestamp --> causal ordered sequence & level index
# for causal attention (beyond pure temporal causality, suitable for GAT)
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