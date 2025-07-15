import glob
import torch
import itertools
import torch.nn as nn
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

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

    def forward(self, idx, target):

        def causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          return causal_mask

        S = idx.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device="cuda", _compile=True)

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

from typing import Optional
import torch 

# Embedding ensemble function - abstraction guidance & generation | execution & planning
def sandwich_embedding(
    token_embeddings: torch.Tensor,        # [B, S, D]
    high_level_embeddings: Optional[torch.Tensor] = None,   # [B, L1, D]
    low_level_embeddings: Optional[torch.Tensor] = None,    # [B, S, D]
    K: int = 1,                      # abstraction ratio
):
    """
    low-level embedding is sliced every K tokens
    passive observation on low-level execution is allowed
    """
    seq_len = token_embeddings.shape[1]

    if high_level_embeddings is not None: 
        L1 = high_level_embeddings.shape[1]
        assert L1 * K <= seq_len < (L1 + 1) * K, f"Execution without purpose or planning without grounding is not allowed: {L1 * K} < {seq_len} <= {(L1 + 1) * K}"
        cond_embeddings = high_level_embeddings.repeat_interleave(K, dim=1)
        token_embeddings[:, :L1 * K] += cond_embeddings
    
    if low_level_embeddings is not None: 
        L2 = low_level_embeddings.shape[1]
        assert seq_len <= L2, f"Planning without grounding is not allowed: {seq_len} > {L2}"
        token_embeddings += low_level_embeddings[:, : seq_len]

    return token_embeddings


# Abstraction model 
