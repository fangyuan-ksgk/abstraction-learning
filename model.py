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
    

# Interestingly, abstract model doesn't require communicaion-like conditioning, instead it uses a temporal aligned embedding ensemble, which makes 
# sense, since all abstract modules eventually work together, in the same time. 
# Remark #1. We deal with simplest case where embedding dimension is the same across abstraction levels. 

# Embedding ensemble function 
# v1. pure addition (AR model) requires left-shift on plan embeddings
# v1. pure addition (diffusion model) aligns embedding temporally
# v2. attention-based (AR model) requires left-shift on plan embeddings
# v2. attention-based (diffusion model) aligns embedding temporally

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
        

def infer_time_step(idx: list, K: int)->int:
    # longest time step provided by the leve-wise sequence
    return max([seq.shape[1] * (K**l) for l, seq in enumerate(idx)])

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

    def compute_loss(self, idx, target, high_embed, low_embed):
        rep, logits = self.forward(idx, high_embed, low_embed)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return rep, loss

# --------------------------------------------------------------------------------------------------------------------------


# Generative Abstraction Model (GAT)
# --------------------------------------------------------------------------------------------------------------------------

class GAT(nn.Module): 

    def __init__(self, config):
        super().__init__()
        self.K = config.K
        self.L = config.L # abstraction level
        self.condgpts = nn.ModuleList([CondGPT(config) for _ in range(config.L)])

    def forward(self, idx, target): 
        """
        TBD: for multi-level sequence, compute loss at each level & ensemble them
        idx: [B, S]
        target: [B, S]
        """
        pass 

    def generate(self, idx: Union[torch.Tensor, list]):
    
        embed_cache = [None for l in range(self.L)]

        def generate_level(l: int, curr: list, t: int, low_embed: Optional[torch.Tensor] = None) -> list: 
            """
            low_level_embeddings is strided to reduce carbon footprint :>
            """
            # print(f"generate level {l} at time {t} x {self.K**l}")

            if l <= self.L:
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
        T = infer_time_step(sequences, self.K)
        for t in range(1, T+2):
            sequences = generate_level(0, sequences, t, None)
                    
        return sequences
            




