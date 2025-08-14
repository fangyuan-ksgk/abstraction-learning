import glob
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional
from collections import defaultdict 
from dataclasses import dataclass, field
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from constant import PLACE_HOLDER_STATE_TOK, PLACE_HOLDER_ACTION_TOK
from utils import (
    get_next_token_level, HierSeq, HierTraj, create_loss_mask,
    make_interleave_embd, create_traj_loss_mask, get_next_traj_token,
    _build_interleave_embd
)


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
    L: int = 4  # total # of levels (including 0-th level)
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

    def forward(self, batch_data: HierSeq):

        input_idx, sample_idx = batch_data.tokens[:-1], batch_data.sample_idx[:-1]

        def sample_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            return causal_mask & sample_mask

        S = input_idx.shape[0]
        block_mask = create_block_mask(sample_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self._create_hseq_embd(batch_data, do_slice=True)

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        loss = self._compute_hseq_loss(x, batch_data)

        return loss
    
    def generate(self, batch_data: HierSeq, parallel: bool = False):

        input_idx, sample_idx = batch_data.tokens, batch_data.sample_idx

        def sample_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            return causal_mask & sample_mask

        S = input_idx.shape[0]
        block_mask = create_block_mask(sample_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self._create_hseq_embd(batch_data, do_slice=False)

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        if parallel:
            batch_data = self._parallel_generate(x, batch_data) # fast, parallel AR generation (partial conditioning with Padding)
        else:
            batch_data = self._causal_generate(x, batch_data) # slow, sequential AR generation (full conditioning)

        return batch_data

    def _create_hseq_embd(self, batch_data: HierSeq, do_slice: bool = True) -> torch.Tensor: 

        if do_slice: 
            input_idx, input_levels = batch_data.tokens[:-1], batch_data.levels[:-1]
        else: 
            input_idx, input_levels = batch_data.tokens, batch_data.levels

        S = input_idx.shape[0]
        x = torch.zeros(1, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L): # per-level embedding
            level_mask = (input_levels == l)
            if level_mask.any():  
                level_tokens = input_idx[level_mask]  
                level_embed = self.wtes[l](level_tokens) + self.level_embeddings[l].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                x[:,level_mask] = level_embed

        return x

    def _compute_hseq_loss(self, x: torch.Tensor, batch_data: HierSeq) -> torch.Tensor: 

        target_idx, target_levels = batch_data.tokens[1:], batch_data.levels[1:]
        sample_idx = batch_data.sample_idx

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

    # Caveat: DAT doesn't have lm_head on level 0, therefore index for lm_head is shrunk by one, lm_head[l-1] is the lm_head for level l 
    #       - GAT, however, has lm_head on every level, therefore the index for level l lm_head is lm_head[l], I copy the code from DAT 
    #       - which led to such hidden bug (which leads to a shift-by-one permutation behaviour). Let's try again to see if it's fixed. 

    def _causal_generate(self, x: torch.Tensor, batch_data: HierSeq): 
        level_groups = batch_data.next_level_groups() 

        for l_next, group in level_groups.items(): 
            batch_indices, masks, timestamps = zip(*group)
            reprs = torch.stack([x[0, mask][-1] for mask in masks])
            next_tokens = torch.argmax(30 * torch.tanh(self.lm_heads[l_next](reprs) / 30), dim=-1)
            for i, b in enumerate(batch_indices): 
                batch_data.insert_tokens(b, next_tokens[i], l_next, timestamps[i])
        
        return batch_data

    def _parallel_generate(self, x: torch.Tensor, batch_data: HierSeq): 
        level_groups = batch_data.get_abstract_groups()  # (level, (batch_indices, mask_positions, timestamps))

        for l_curr, (batch_indices, mask_positions, timestamps) in level_groups.items(): 
            reprs = x[0, mask_positions] 
            new_tokens = torch.argmax(30 * torch.tanh(self.lm_heads[l_curr](reprs) / 30), dim=-1)
            for i, (b, t) in enumerate(zip(batch_indices, timestamps)): 
                batch_data.insert_tokens(b, new_tokens[i].unsqueeze(0), l_curr, t.unsqueeze(0))

        return batch_data


# --------------------------------------------------------------------------------------------------------------------------



# It's hard to believe how much things are complicated when I try to build a GAT module for RL Agent. 
# Abstract Policy Transformer (APT) for Snake Game and more
# --------------------------------------------------------------------------------------------------------------------------

@dataclass
class DATConfig:
    n_layer : int = 4
    n_head : int = 2
    n_embd : int = 64
    flex_kernel_options: Optional[dict] = None
    K: int = 4  # abstraction ratio
    L: int = 3  # total # of levels (including 0-th level)
    vocab_size_list: list = field(default_factory=lambda: [64, 32]) # vocab for abstractions
    device: str = "cuda"
    _compile: bool = True
    level_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


# Abstract Decision transformer (DAT)
class DAT(nn.Module):

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
    

    def forward(self, batch_data: HierSeq, trajectories: list):

        input_levels = batch_data.levels[:-1]
        sample_idx = batch_data.sample_idx

        def markov_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            pos = torch.arange(len(input_levels), device=input_levels.device)
            markov_mask = (input_levels[kv_idx] == 0) & torch.any(
                (pos > kv_idx) & (pos < q_idx) & (input_levels == 0))
            return causal_mask & sample_mask & ~markov_mask

        S = input_levels.shape[0]
        block_mask = create_block_mask(markov_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self._create_htraj_embd(batch_data, trajectories)

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        loss = self._compute_htraj_loss(x, batch_data, trajectories)

        return loss

    def generate(self, batch_data: HierSeq, trajectories: list):
        """
        Note: trajectories w/o reward is newly generated
        """

        idx = batch_data.tokens
        levels = batch_data.levels
        timestamps = batch_data.timestamps
        sample_idx = batch_data.sample_idx

        def markov_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            sample_mask = sample_idx[q_idx] == sample_idx[kv_idx]
            pos = torch.arange(len(levels), device=levels.device)
            markov_mask = (levels[kv_idx] == 0) & torch.any(
                (pos > kv_idx) & (pos < q_idx) & (levels == 0))
            return causal_mask & sample_mask & ~markov_mask

        S = levels.shape[0]
        block_mask = create_block_mask(markov_causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self._create_htraj_embd(batch_data, trajectories, do_slice=False)

        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        batch_data, trajectories = self._hiearchical_generate(x, batch_data, trajectories)

        return batch_data, trajectories 

    def act(self, batch_data: HierSeq, trajectories: list): 

        new_action = self._get_new_action(trajectories) 
        while not new_action:   
            batch_data, trajectories = self.generate(batch_data, trajectories)
            new_action = self._get_new_action(trajectories)

        return new_action
        
    def _get_new_action(self, trajectories: list) -> list:
        pairs = [] # (sample_idx, action_idx)
        for b, trajectory in enumerate(trajectories):
            _, action, reward = trajectory
            action_size = action.size(0) if action is not None else 0
            reward_size = reward.size(0) if reward is not None else 0
            n_new_action = action_size - reward_size
            if n_new_action > 0:
                new_action = action[-n_new_action:]
                pairs.append((b, new_action))
        return pairs

    def _embd_trajectory(self, batch_data: HierTraj, trajectories: list) -> torch.Tensor:
        """ 
        (a). [s,a] (b). [a,s] (c). [s,a,s] (d). [a,s,a]
        """
        embds = [] 
        for b, trajectory in enumerate(trajectories):

            s_embd = self.state_encoder(trajectory[0])
            a_embd = self.action_encoder(trajectory[1])

            sample_l0_mask = torch.logical_and(batch_data.levels == 0, batch_data.sample_idx == b)
            first_tok = batch_data.tokens[sample_l0_mask][0].item()
            ft_act = first_tok == PLACE_HOLDER_ACTION_TOK

            traj_embd = _build_interleave_embd(s_embd, a_embd, ft_act)
            embds.append(traj_embd)

        return torch.cat(embds, dim=0).unsqueeze(0)                


    def _create_htraj_embd(self, batch_data: HierTraj, trajectories: list, do_slice: bool = True) -> torch.Tensor: 
 
        traj_embds = self._embd_trajectory(batch_data, trajectories)

        if do_slice: 
            input_levels = batch_data.levels[:-1]
            input_idx = batch_data.tokens[:-1]
            if batch_data.levels[-1] == 0:
                traj_embds = traj_embds[:, :-1]
        else: 
            input_levels, input_idx = batch_data.levels, batch_data.tokens

        S = input_idx.shape[0]
        x = torch.zeros(1, S, self.level_embeddings.shape[1], device=self.device)

        for l in range(self.L):
            level_mask = (input_levels == l)

            if level_mask.any():  
                if l == 0: 
                    x[:, level_mask] = traj_embds # embedding replacement
                else: 
                    level_tokens = input_idx[level_mask]  
                    level_embed = self.wtes[l-1](level_tokens) + self.level_embeddings[l-1].unsqueeze(0)  # (num_tokens, n_embd) + (1, n_embd)
                    x[:,level_mask] = level_embed

        return norm(x)


    def _compute_htraj_loss(self, x: torch.Tensor, batch_data: HierTraj, trajectories: list) -> torch.Tensor:
   
        action_tensor = torch.cat([t[1] for t in trajectories], dim=0)
        state_tensor = torch.cat([t[0] for t in trajectories], dim=0)

        total_loss = 0.0 
        total_weight = 0.0
        loss_mask, loss_mask_state, loss_mask_action = create_traj_loss_mask(batch_data)

        for l in range(self.L): # per-level projection
            mask = (batch_data.levels[1:] == l) & loss_mask
            
            if mask.any():
                if l == 0: 
                    action_select_loss = self.action_decoder.forward(
                        x[0, mask & batch_data.action_mask[1:]],
                        action_tensor[loss_mask_action]
                    )
                    state_predict_loss = self.state_decoder.forward(
                        x[0, mask & batch_data.state_mask[1:]],
                        state_tensor[loss_mask_state]
                    )
                    level_loss = action_select_loss + state_predict_loss
                else: 
                    level_logits = self.lm_heads[l-1](x[:, mask])
                    level_logits = 30 * torch.tanh(level_logits / 30).float()                
                    level_loss = F.cross_entropy(
                        level_logits.view(-1, level_logits.size(-1)),
                        batch_data.tokens[1:][mask]
                    )
                    total_loss += self.level_weights[l] * level_loss
                    total_weight += self.level_weights[l]
        
        return total_loss / total_weight
 
    def _group_by_next_level(self, batch_data: HierTraj):
        level_groups = defaultdict(list) 
        for b in range(batch_data.batch_size): 
            mask = batch_data.sample_idx == b
            l_next, t_next, tok_next = get_next_traj_token(
                batch_data.levels[mask], batch_data.timestamps[mask], batch_data.tokens[mask], self.K, self.L
            )
            level_groups[l_next.item()].append((b, mask, t_next, tok_next))
        return level_groups

    def _generate_tokens_by_type(self, x: torch.Tensor, masks: list, toks_next: list, token_type: int, decoder: nn.Module):
        filtered_masks = [mask for mask, tok in zip(masks, toks_next) if tok == token_type]
        if not filtered_masks:
            return None
        x_len = x.shape[1]
        reprs = torch.stack([x[0, mask[:x_len]][-1] for mask in filtered_masks])
        return decoder.generate(reprs)
        
    def _process_zero_level_tokens(self, x: torch.Tensor, group: list, batch_data: HierTraj, trajectories: list):

        batch_indices, masks, timestamps, toks_next = zip(*group)
        
        actions = self._generate_tokens_by_type(x, masks, toks_next, PLACE_HOLDER_ACTION_TOK, self.action_decoder)
        states = self._generate_tokens_by_type(x, masks, toks_next, PLACE_HOLDER_STATE_TOK, self.state_decoder)

        i_a = i_s = 0
        for i, b in enumerate(batch_indices): 
            if toks_next[i] == PLACE_HOLDER_ACTION_TOK: 
                new_action = actions[i_a:i_a + 1] if trajectories[b][1] is None else torch.cat([trajectories[b][1], actions[i_a:i_a+1]])
                trajectories[b] = (trajectories[b][0], new_action, trajectories[b][2])
                batch_data.insert_next_token(b, PLACE_HOLDER_ACTION_TOK, 0, timestamps[i])
                i_a += 1
            elif toks_next[i] == PLACE_HOLDER_STATE_TOK: 
                new_state = states[i_s:i_s + 1] if trajectories[b][0] is None else torch.cat([trajectories[b][0], states[i_s:i_s+1]])
                trajectories[b] = (new_state, trajectories[b][1], trajectories[b][2])
                batch_data.insert_next_token(b, PLACE_HOLDER_STATE_TOK, 0, timestamps[i])
                i_s += 1
            else: 
                raise ValueError(f"Invalid token: {toks_next[i]}")

    def _process_higher_level_tokens(self, x: torch.Tensor, group: list, batch_data: HierTraj, l_next: int):

        batch_indices, masks, timestamps, _ = zip(*group)
        
        reprs = torch.stack([x[0, mask][-1] for mask in masks])
        tokens = torch.argmax(30 * torch.tanh(self.lm_heads[l_next-1](reprs) / 30), dim=-1)
        
        for i, b in enumerate(batch_indices): 
            batch_data.insert_next_token(b, tokens[i], l_next, timestamps[i])


    def _hiearchical_generate(self, x: torch.Tensor, batch_data: HierTraj, trajectories: list):

        level_groups = self._group_by_next_level(batch_data)
        
        for l_next, group in level_groups.items(): 
            if l_next == 0: 
                self._process_zero_level_tokens(x, group, batch_data, trajectories)
            else: 
                self._process_higher_level_tokens(x, group, batch_data, l_next)

        return batch_data, trajectories



# TBD: DRA loss computation (using reward signal)
# --------------------------------------------------------------------------------------------------------------------------
