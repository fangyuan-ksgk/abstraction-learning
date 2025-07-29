import glob
import torch
import itertools
import torch.nn as nn
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from constant import BOS_TOKEN_ID
from utils import get_next_token_level, BatchedHierSeq, create_loss_mask, get_ext_embds


# GPT 
# --------------------------------------------------------------------------------------------------------------------------

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
    
# --------------------------------------------------------------------------------------------------------------------------


# (I). Attention pattern: Hiearchical Causality 
# (II). GAT: GPT with level-wise WTE & LM head
# GAT for language modeling
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

    def forward(self, batch_data: BatchedHierSeq):

        input_idx = batch_data.tokens[:-1]
        target_idx = batch_data.tokens[1:]
        input_levels = batch_data.levels[:-1]
        target_levels = batch_data.levels[1:]

        sample_idx = batch_data.sample_idx

        def sample_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            return causal_mask & sample_mask

        S = input_idx.shape[0]
        block_mask = create_block_mask(sample_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)


        x = torch.zeros(1, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L): # per-level embedding
            level_mask = (input_levels == l)
            if level_mask.any():  
                level_tokens = input_idx[level_mask]  
                level_embed = self.wtes[l](level_tokens) + self.level_embeddings[l].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                x[:,level_mask] = level_embed

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        total_loss = 0.0 
        total_weight = 0.0
        loss_mask = create_loss_mask(sample_idx)[1:]

        for l in range(self.L): # per-level projection
            target_level_mask = (target_levels == l) & loss_mask
            if target_level_mask.any():  
                level_logits = self.lm_heads[l](x[:, target_level_mask])
                level_logits = 30 * torch.tanh(level_logits / 30).float()                
                level_loss = F.cross_entropy(
                    level_logits.view(-1, level_logits.size(-1)),
                    target_idx[target_level_mask]
                )
                total_loss += self.level_weights[l] * level_loss
                total_weight += self.level_weights[l]

        return total_loss / total_weight

    
    def generate(self, batch_data: BatchedHierSeq):

        input_idx, input_levels, input_timestamps = batch_data.tokens, batch_data.levels, batch_data.timestamps
        sample_idx = batch_data.sample_idx

        def sample_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            return causal_mask & sample_mask

        S = input_idx.shape[0]
        block_mask = create_block_mask(sample_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = torch.zeros(1, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L): # per-level embedding
            level_mask = (input_levels == l)
            if level_mask.any():  
                level_tokens = input_idx[level_mask]  
                level_embed = self.wtes[l](level_tokens) + self.level_embeddings[l].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                x[:,level_mask] = level_embed

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        for b in range(batch_data.batch_size):

            mask = sample_idx == b
            l_next, t_next = get_next_token_level(input_levels[mask], input_timestamps[mask], self.K, self.L)

            logits = self.lm_heads[l_next](x[0, mask][-1])
            logits = 30 * torch.tanh(logits / 30).float()
            next_token = torch.argmax(logits, dim=-1)

            batch_data.insert_next_token(b, next_token, l_next, t_next)

        return batch_data

# --------------------------------------------------------------------------------------------------------------------------


# Abstract Policy Transformer (APT) for Snake Game and more
# --------------------------------------------------------------------------------------------------------------------------

@dataclass
class APTConfig:
    n_layer : int = 4
    n_head : int = 2
    n_embd : int = 64
    flex_kernel_options: Optional[dict] = None
    K: int = 4  # abstraction ratio
    L: int = 2  # number of abstraction levels
    vocab_size_list: list = field(default_factory=lambda: [64, 32]) # vocab for abstractions
    device: str = "cuda"
    _compile: bool = True
    level_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


# Abstract policy transformer (APT)
class APT(nn.Module):

    def __init__(
            self, 
            config,
            state_encoder,
            state_decoder,
            action_encoder,
            action_decoder,
        ):
        super().__init__()
        self.num_layers = config.n_layer
        self.L = config.L
        self.K = config.K

        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        
        # abstract token encoder & decoder
        self.wtes = nn.ModuleList([nn.Embedding(vocab_size, config.n_embd) for vocab_size in config.vocab_size_list])
        self.lm_heads = nn.ModuleList([CastedLinear(config.n_embd, vocab_size) for vocab_size in config.vocab_size_list])
        for lm_head in self.lm_heads:
            lm_head.weight.data.zero_()
        
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.level_embeddings = nn.Parameter(torch.randn(config.L-1, config.n_embd))

        self.device = config.device
        self._compile = config._compile
        self.level_weights = config.level_weights


    # Issue: state & action are not 'indexed' (action usually is but state isn't), we need to predict both action & next-state
    # Progress: 
    # - (Done) Add Markov execution mask (0-th level token has window of 1)
    # - (Done) Add raw State & Action tensor as input (for loss computation) 
    # - (TBD) DRA using reward signal

    def _get_ext_embds(self, trajectories): 
        ext_embds = []
        for trajectory in trajectories:
            state_embd = self.state_encoder(trajectory[0])
            action_embd = self.action_encoder(trajectory[1])
            extended_embd = torch.cat([action_embd, state_embd], dim=1)
            ext_embds.append(extended_embd)
        return torch.cat(ext_embds, dim=0).unsqueeze(0)

    def _create_hierarchical_embeddings(self, batch_data, trajectories):
 
        input_levels = batch_data.levels[:-1]
        input_idx = batch_data.tokens[:-1]
        S = input_idx.shape[0]
        
        x = torch.zeros(1, S, self.level_embeddings.shape[1], device=self.device)
        ext_embds = self._get_ext_embds(trajectories)
        
        for l in range(self.L):
            level_mask = (input_levels == l)
            if level_mask.any():  
                if l == 0: 
                    x[:, level_mask] = ext_embds  # External embeddings!
                else: 
                    level_tokens = input_idx[level_mask]  
                    level_embed = (self.wtes[l-1](level_tokens) + 
                                self.level_embeddings[l-1].unsqueeze(0))
                    x[:, level_mask] = level_embed
        
        return norm(x)

    def _compute_hierarchical_loss(self, hidden_states, batch_data, trajectories):
   
        target_levels = batch_data.levels[1:]
        target_idx = batch_data.tokens[1:]
        loss_mask = create_loss_mask(batch_data.sample_idx)[1:]
        
        total_loss = 0.0 
        total_weight = 0.0
        
        for l in range(self.L):
            target_level_mask = (target_levels == l)
            if target_level_mask.any():
                
                if l == 0:  # Special handling for level 0
                    loss_mask_l0 = loss_mask[target_level_mask]
                    action_tensor = torch.cat([t[1] for t in trajectories], dim=0)
                    state_tensor = torch.cat([t[0] for t in trajectories], dim=0)
                    
                    action_loss = self.action_decoder.forward(
                        hidden_states[:, target_level_mask][0, loss_mask_l0],
                        action_tensor[1:][loss_mask_l0]
                    )
                    state_loss = self.state_decoder.forward(
                        hidden_states[:, target_level_mask][0, loss_mask_l0],
                        state_tensor[1:][loss_mask_l0]
                    )
                    level_loss = action_loss + state_loss
                    
                else:  # Standard language modeling for other levels
                    target_level_mask = target_level_mask & loss_mask
                    level_logits = self.lm_heads[l-1](hidden_states[:, target_level_mask])
                    level_logits = 30 * torch.tanh(level_logits / 30).float()                
                    level_loss = F.cross_entropy(
                        level_logits.view(-1, level_logits.size(-1)),
                        target_idx[target_level_mask]
                    )
                    
                total_loss += self.level_weights[l] * level_loss
                total_weight += self.level_weights[l]
        
        return total_loss / total_weight
    

    def forward(self, batch_data: BatchedHierSeq, trajectories: list):

        input_idx = batch_data.tokens[:-1]
        target_idx = batch_data.tokens[1:]
        input_levels = batch_data.levels[:-1]
        target_levels = batch_data.levels[1:]

        sample_idx = batch_data.sample_idx

        def markov_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            
            pos = torch.arange(len(input_levels), device=input_levels.device)
            markov_mask = (input_levels[kv_idx] == 0) & torch.any(
                (pos > kv_idx) & (pos < q_idx) & (input_levels == 0))
            
            return causal_mask & sample_mask & ~markov_mask

        S = input_idx.shape[0]
        block_mask = create_block_mask(markov_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        ext_embds = self._get_ext_embds(trajectories)

        x = torch.zeros(1, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L): # per-level embedding
            level_mask = (input_levels == l)

            if level_mask.any():  
                if l == 0: 
                    x[:, level_mask] = ext_embds # place-holder tokens embedding replacement
                else: 
                    level_tokens = input_idx[level_mask]  
                    level_embed = self.wtes[l-1](level_tokens) + self.level_embeddings[l-1].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                    x[:,level_mask] = level_embed

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        total_loss = 0.0 
        total_weight = 0.0
        loss_mask = create_loss_mask(sample_idx)[1:]

        for l in range(self.L): # per-level projection
            target_level_mask = (target_levels == l)
            if target_level_mask.any():

                if l == 0: 
                    loss_mask_l0 = loss_mask[target_level_mask]

                    action_tensor = torch.cat([t[1] for t in trajectories], dim=0)
                    state_tensor = torch.cat([t[0] for t in trajectories], dim=0)

                    action_select_loss = self.action_decoder.forward(
                        x[:, target_level_mask][0, loss_mask_l0],
                        action_tensor[1:][loss_mask_l0]
                    )
                    state_predict_loss = self.state_decoder.forward(
                        x[:, target_level_mask][0, loss_mask_l0],
                        state_tensor[1:][loss_mask_l0]
                    )
                    
                    level_loss = action_select_loss + state_predict_loss
                else: 
                    target_level_mask = target_level_mask & loss_mask
                    level_logits = self.lm_heads[l-1](x[:, target_level_mask])
                    level_logits = 30 * torch.tanh(level_logits / 30).float()                
                    level_loss = F.cross_entropy(
                        level_logits.view(-1, level_logits.size(-1)),
                        target_idx[target_level_mask]
                    )
                total_loss += self.level_weights[l] * level_loss
                total_weight += self.level_weights[l]

        return total_loss / total_weight

    
    def generate(self, batch_data: BatchedHierSeq):

        input_idx, input_levels, input_timestamps = batch_data.tokens, batch_data.levels, batch_data.timestamps
        sample_idx = batch_data.sample_idx

        def sample_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            return causal_mask & sample_mask

        S = input_idx.shape[0]
        block_mask = create_block_mask(sample_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = torch.zeros(1, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L): # per-level embedding
            level_mask = (input_levels == l)
            if level_mask.any():  
                level_tokens = input_idx[level_mask]  
                level_embed = self.wtes[l](level_tokens) + self.level_embeddings[l].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                x[:,level_mask] = level_embed

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        for b in range(batch_data.batch_size):

            mask = sample_idx == b
            l_next, t_next = get_next_token_level(input_levels[mask], input_timestamps[mask], self.K, self.L)

            logits = self.lm_heads[l_next](x[0, mask][-1])
            logits = 30 * torch.tanh(logits / 30).float()
            next_token = torch.argmax(logits, dim=-1)

            batch_data.insert_next_token(b, next_token, l_next, t_next)

        return batch_data







# --------------------------------------------------------------------------------------------------------------------------