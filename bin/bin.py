import glob
import torch
import itertools
import torch.nn as nn
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from constant import BOS_TOKEN_ID

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        assert x.ndim == 4 # multihead attention
        d = x.shape[3]//2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, flex_kernel_options=None):
        super().__init__()
        assert dim % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.dim = dim
        self.n_head = n_head        
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        # value residual lambda 
        self.lamb = nn.Parameter(torch.tensor(0.5))  # @Grad62304977
        # rotary embeddings
        self.rotary = Rotary(dim // n_head)
        # output projection
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        # flex attention kernel options
        self.flex_kernel_options = flex_kernel_options

    def forward(self, x, v1=None, block_mask=None):
        B, T = x.size(0), x.size(1)  
        # Compute Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)        
        if v1 is None:
            v1 = v  # If this is the first block, set v1 to v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)  # @Grad62304977
        q, k = norm(q), norm(k) # QK norm suggested by @Grad62304977
        q, k = self.rotary(q), self.rotary(k)        
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            kernel_options=self.flex_kernel_options
        )
        y = y.transpose(1, 2).contiguous().view_as(x)       
        y = self.c_proj(y)
        return y, v1


class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.flex_kernel_options)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(norm(x), v1, block_mask)
        x = x + x1
        x = x + self.mlp(norm(x))
        return x, v1
    

# Remove comm_idx since abstract model will process comm inside
# - external input is just a sequence of tokens (in language modeling case)
@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    eoc_idx : int = 5826319 # Issue #1. This is much larger than vocab_size (funny mistake, where is the tokenizer?)
    eos_idx : int = 5826320
    K: int = 4  # abstraction ratio
    L: int = 3  # number of abstraction levels
    device: str = "cuda"
    _compile: bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.n_layer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()
        self.device = config.device
        self._compile = config._compile

    def forward(self, idx, target):

        def causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          return causal_mask

        S = idx.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        # forward the GPT model itself
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss
    

# Ver.1 (Discarded): Hiearchical generation is not an architecture design, but an attention pattern
#                  - Hiearchical architecture with level-wise generation is temporally asynchronous, propagation can't be paralelized, and attention cross level is hard-to-implement, too
#                 -> Trapped in thoughts on how to compromise the causal connection which is crucial for integrated information, abandoned. 
#              seq_of_tokens --> flatten --> hiearchical causal connection is just a special attention pattern --> one transformer blocks to take care of causal connection, with level-wise WTE & LM head suffices 

from typing import Optional, Union
import torch 

# Utility function
# --------------------------------------------------------------------------------------------------------------------------

def is_valid_embed(embed: Optional[torch.Tensor]) -> bool:
    return embed is not None and embed.shape[1]>0

def prep_idx(idx: Union[torch.Tensor, list], L: int) -> list:

    if isinstance(idx, torch.Tensor): # level-0 sequence
        B = idx.shape[0] if len(idx.shape) > 1 else 1
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
        empty_tensor = torch.zeros((B, 0), device=idx.device, dtype=idx.dtype)
        return [idx] + [empty_tensor for _ in range(1, L)]

    else:  # all-level sequences (list of tensors)
        assert isinstance(idx, list) and len(idx) == L, f"Missing sequence for {L} abstraction levels, currently only got {len(idx)}."
        ref_seq = next((s for s in idx if isinstance(s, torch.Tensor) and s.numel() > 0), None)
        assert ref_seq is not None, "No valid reference sequence found to determine device and dtype"
        device, dtype = ref_seq.device, ref_seq.dtype
        B = ref_seq.shape[0] if len(ref_seq.shape) > 1 else 1
        empty_tensor = torch.zeros((B, 0), device=device, dtype=dtype)
        
        sequences = []
        for seq in idx: 
            assert isinstance(seq, torch.Tensor), "All sequences must be tensors"
            if seq.numel() == 0:
                seq = empty_tensor
            else:
                seq = seq.unsqueeze(0) if len(seq.shape) == 1 else seq
            sequences.append(seq)

        return sequences

def infer_time_step(idx: list, K: int, mode: str = "max")->int:
    """
    Use max time for generation (interpret past + next step), min time for training (full conditions)
    """
    if mode == "max":
        return max([seq.shape[1] * (K**l) for l, seq in enumerate(idx)])
    elif mode == "min":
        return min([seq.shape[1] * (K**l) for l, seq in enumerate(idx)])
    else:
        raise ValueError(f"Invalid mode: {mode}")

# --------------------------------------------------------------------------------------------------------------------------



# Sandwich embedding: incorporate planning guidance & grounding
# --------------------------------------------------------------------------------------------------------------------------

def sandwich_embedding(
    low_embed: Optional[torch.Tensor] = None,    # [B, S1, D]
    tok_embed: Optional[torch.Tensor] = None,    # [B, S, D]
    high_embed: Optional[torch.Tensor] = None,   # [B, S2, D]
    K: int = 1,                      # abstraction ratio
    ):
    """
    sandwich_i = tok[i-1] + low[i] + high[(i-1)//K] for i in [0, 1, ..., S]
    """
    assert is_valid_embed(tok_embed) or is_valid_embed(low_embed), "tok_embed or low_embed must be provided"

    if not is_valid_embed(tok_embed): # generation mode, first abstract token
        assert is_valid_embed(low_embed), "When tok_embed is None, low_embed must be provided"
        return low_embed[:, :1]
    
    S = tok_embed.shape[1]
    
    if is_valid_embed(high_embed): # planning guidance
        S2 = high_embed.shape[1]
        tok_embed[:, :min(S, K * S2)] += high_embed.repeat_interleave(K, dim=1)[:, :min(S, K * S2)]
    
    if is_valid_embed(low_embed): # grounding
        S1 = low_embed.shape[1]
        tok_embed = torch.cat([torch.zeros(*tok_embed.shape[:1], 1, tok_embed.shape[2], device=tok_embed.device, dtype=tok_embed.dtype), tok_embed], dim=1)
        tok_embed[:, :min(S1, S + 1)] += low_embed[:, 0:min(S1, S + 1)]

    return tok_embed


# Conditional GPT takes in higher & lower embedding conditioning
# TBD: - include KV-cache
# --------------------------------------------------------------------------------------------------------------------------

class CondGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.n_layer
        self.K = config.K
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()
        self.device = config.device
        self._compile = config._compile

    def forward(self, idx, high_embed, low_embed): 

        x = self.transformer.wte(idx)
        x = sandwich_embedding(low_embed, x, high_embed, self.K) # one-liner
        x = norm(x)
        x0 = x
        v1 = None

        def causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          return causal_mask

        S = x.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)                              
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        return x, logits

    def generate(self, idx, high_embed, low_embed):
        rep, logits = self.forward(idx, high_embed, low_embed)
        idx_pred = torch.argmax(logits[:, -1, :], dim=-1)
        return rep, idx_pred

    def compute_loss(self, idx, target, high_embed, low_embed, weight):
        rep, logits = self.forward(idx, high_embed, low_embed)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
        loss = (loss * weight.repeat_interleave(logits.shape[0], dim=0)).mean()
        return rep, loss

# --------------------------------------------------------------------------------------------------------------------------




# Generative Abstraction Model (GAT)
# --------------------------------------------------------------------------------------------------------------------------

@dataclass
class GATConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    eoc_idx : int = 5826319 
    eos_idx : int = 5826320
    K: int = 4  # abstraction ratio
    L: int = 3  # number of abstraction levels
    device: str = "cuda"
    _compile: bool = True
    level_weights: Optional[list] = None
    w_cond: float = 1.0
    w_uncond: float = 1.0

class GAT(nn.Module): 

    def __init__(self, config):
        super().__init__()
        self.K = config.K
        self.L = config.L # abstraction level
        self.level_weights = config.level_weights if config.level_weights is not None else [1.0] * config.L
        self.w_cond, self.w_uncond = config.w_cond, config.w_uncond
        self.condgpts = nn.ModuleList([CondGPT(config) for _ in range(config.L)])

    def forward(self, idx: list): 
        """
        Un-weighted ver. of loss ensemble
        TBD: add-in weight for different levels, conditioned-based & condition-free loss etc.
        """

        assert len(idx) == self.L, f"Missing sequence for {self.L} abstraction levels, currently only got {len(idx)}."
        embed_cache = [None for l in range(self.L)]

        def compute_loss_level(l: int, curr_loss: torch.Tensor, t: int, low_embed: Optional[torch.Tensor] = None):
            print(f"  : compute loss level {l} at time {t} x {self.K**l}")

            if l < self.L:
                n_tok = idx[l].size(1)
                assert n_tok >= t, f"Missing tokens at level {l} at time {t * self.K**l}"
                weights = self.get_weights(l, t, idx[l])

                _, mixed_loss_l = self.condgpts[l].compute_loss(
                    idx[l][:, :-1],
                    idx[l][:, 1:] if l == 0 else idx[l],
                    embed_cache[l+1] if l < self.L-1 else None,
                    low_embed,
                    weights[1:] if l == 0 else weights
                )

                embeds, _ = self.condgpts[l].forward(
                    idx[l][:, :t],
                    embed_cache[l+1] if l < self.L-1 else None,
                    low_embed
                )

                curr_loss += mixed_loss_l

                if l > 0: 
                    embed_cache[l] = embeds

                if t % self.K == 0: 
                    return compute_loss_level(l+1, curr_loss, t // self.K, embeds[:, ::self.K])

            return curr_loss

        T = infer_time_step(idx, self.K, mode="min")
        loss = torch.tensor(0.0, device=idx[0].device)

        for t in range(self.K, T + self.K, self.K): 
            loss = compute_loss_level(0, loss, min(t, T), None)

        return loss
        

    def generate(self, idx: Union[torch.Tensor, list]):
    
        embed_cache = [None for l in range(self.L)]

        def generate_level(l: int, curr: list, t: int, low_embed: Optional[torch.Tensor] = None) -> list: 
            """
            low_level_embeddings is strided to reduce carbon footprint :>
            """
            # print(f"  : generate level {l} at time {t} x {self.K**l}")

            if l < self.L:
                n_tok = curr[l].size(1)
                if n_tok < t:
                    assert n_tok == t-1, f"Token counts {n_tok} mismatch with # of step {t-1}"
                    _, new_idx = self.condgpts[l].generate(
                                        curr[l][:, :t-1], 
                                        embed_cache[l+1] if l < self.L-1 else None, 
                                        low_embed
                                    )
                    curr[l] = torch.cat([curr[l], new_idx.unsqueeze(1)], dim=1)
                
                embeddings, _ = self.condgpts[l].forward(
                        curr[l][:, :t], 
                        embed_cache[l+1] if l < self.L-1 else None, 
                        low_embed
                        )
                    
                if l > 0: 
                    embed_cache[l] = embeddings
                
                if t % self.K == 0: 
                    return generate_level(l+1, curr, t // self.K, embeddings[:, ::self.K]) # only pass strided embedding to save memory
            
            return curr

        sequences = prep_idx(idx, self.L)
        T = infer_time_step(sequences, self.K, mode="max")
        for t in range(1, T+2):
            sequences = generate_level(0, sequences, t, None)
                    
        return sequences

    def get_weights(self, l: int, t: int, idx: torch.Tensor): 
        S = idx.shape[1] if idx.ndim == 2 else idx.shape[0]
        weights = torch.full((S,), self.w_uncond * self.level_weights[l], dtype=idx.dtype, device=idx.device)
        weights[:t] = self.w_cond * self.level_weights[l]
        return weights
            
# --------------------------------------------------------------------------------------------------------------------------

# Ver. Jul 27.

@dataclass
class GATConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    eoc_idx : int = 5826319 
    eos_idx : int = 5826320
    K: int = 4  # abstraction ratio
    L: int = 3  # number of abstraction levels
    vocab_size_list: list = field(default_factory=lambda: [128, 64, 32])
    device: str = "cuda"
    _compile: bool = True
    level_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


class GAT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.n_layer
        self.L = config.L
        self.K = config.K
        self.seqflat = SeqFlat(config.K, config.L)
        self.wtes = nn.ModuleList([nn.Embedding(vocab_size, config.n_embd) for vocab_size in config.vocab_size_list])
        self.lm_heads = nn.ModuleList([CastedLinear(config.n_embd, vocab_size) for vocab_size in config.vocab_size_list])
        for lm_head in self.lm_heads:
            lm_head.weight.data.zero_()
        
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.level_embeddings = nn.Parameter(torch.randn(config.L, config.n_embd))

        self.device = config.device
        self._compile = config._compile
        self.level_weights = config.level_weights

    def forward(self, idx_seq, t_seq=None):
    
        idx, levels, timestamps = self.seqflat(idx_seq, t_seq)
        input_idx = idx[:, :-1]      
        target_idx = idx[:, 1:]      
        input_levels = levels[:, :-1]    
        target_levels = levels[:, 1:]    

        def causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            return causal_mask

        B, S = input_idx.shape
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)        
        x = torch.zeros(B, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L): # per-level embedding
            level_mask = (input_levels == l) 
            if level_mask.any():  
                level_tokens = input_idx[level_mask]  
                level_embed = self.wtes[l](level_tokens) + self.level_embeddings[l].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                x[level_mask] = level_embed

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)
        
        total_loss = 0.0 
        total_weight = 0.0
        
        for l in range(self.L): # per-level projection
            target_level_mask = (target_levels == l) 
            if target_level_mask.any():  
                level_logits = self.lm_heads[l](x[target_level_mask])  
                level_logits = 30 * torch.tanh(level_logits / 30).float()                
                level_loss = F.cross_entropy(level_logits, target_idx[target_level_mask])
                total_loss += self.level_weights[l] * level_loss
                total_weight += self.level_weights[l]
        
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        return total_loss

    
    def generate(self, idx_seq, t_seq=None):

        input_idx, input_levels, input_timestamps = self.seqflat(idx_seq, t_seq)
        
        def causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            return causal_mask

        B, S = input_idx.shape
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)
        x = torch.zeros(B, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L): # per-level embedding
            level_mask = (input_levels == l) 
            if level_mask.any():  
                level_tokens = input_idx[level_mask]  
                level_embed = self.wtes[l](level_tokens) + self.level_embeddings[l].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                x[level_mask] = level_embed

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)
        
        # Batch-size 1 case
        l_next, t_next = get_next_token_level(input_levels, input_timestamps, self.K, self.L)
        logits = self.lm_heads[l_next](x[0, -1:])
        logits = 30 * torch.tanh(logits / 30).float()
        next_token = torch.argmax(logits, dim=-1).squeeze(0)

        idx_seq, t_seq = update_idx_seq(idx_seq, t_seq, next_token, l_next, t_next)
        return idx_seq, t_seq

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