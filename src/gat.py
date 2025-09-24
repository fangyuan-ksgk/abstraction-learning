from src.model import Block, CastedLinear, create_block_mask
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
from src.model import norm
import torch.nn as nn
from src.utils import infer_level


@dataclass
class GATConfig:
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    memory_span : int = 1024 # memory span (for trajectory-level memories)
    K: int = 4  # abstraction ratio
    L: int = 4  # total # of levels (including 0-th level)
    vocab_size_list: list = field(default_factory=lambda: [128, 64, 32])
    device: str = "cuda"
    _compile: bool = True


# New version, aligned with GPT architecture, without level-embedding
class GAT(nn.Module): 

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.n_layer
        self.L = config.L
        self.K = config.K
        self.memory_span = config.memory_span # finest memory span

        # multi-level vocab specific parameters
        self.vocab_sizes = torch.tensor(config.vocab_size_list, device=config.device) + 1
        self.total_vocab_size = sum(self.vocab_sizes)
        self.level_mask_tokens = self.vocab_sizes.cumsum(dim=0) - 1
        self.level_vocab_starts = torch.concat([torch.tensor([0.], device=config.device), self.vocab_sizes.cumsum(dim=0)])[:-1]
        self.level_vocab_ends = self.vocab_sizes.cumsum(dim=0)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.total_vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, self.total_vocab_size)
        self.lm_head.weight.data.zero_()

        self.device = config.device
        self._compile = config._compile

    def forward(self, idx: torch.Tensor, target: torch.Tensor):

        levels = infer_level(idx, self.vocab_sizes, self.level_mask_tokens[0])

        def causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx

            is_higher_level = levels[b, kv_idx] > 0
            is_recent = (q_idx - kv_idx) <= self.memory_span
            keep_mask = is_higher_level | is_recent 
            return causal_mask & keep_mask


        S = idx.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1, _ = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
        loss = loss.view(idx.shape[0], idx.shape[1])
        loss[torch.isin(target, self.level_mask_tokens)] = 0 # ignore loss for mask & pad tokens
        return loss


    def denoise(self, idx: torch.Tensor, denoise_mask: torch.Tensor, denoise_levels: torch.Tensor, temperature: float = 0.0): 

        levels = infer_level(idx, self.vocab_sizes, self.level_mask_tokens[0])

        def causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            
            is_higher_level = levels[b, kv_idx] > 0
            is_recent = (q_idx - kv_idx) <= self.memory_span
            keep_mask = is_higher_level | is_recent 
            return causal_mask & keep_mask

        S = idx.shape[1]
        block_mask = create_block_mask(causal_mask, None, None, S, S, device=self.device, _compile=self._compile)

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        v1 = None

        for i in range(self.num_layers):
            x, v1, _ = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        rep_mask = torch.roll(denoise_mask, -1, dims=1)
        next_token = self._decode(self.lm_head(x[rep_mask]), levels=denoise_levels, temperature=temperature)
        idx[denoise_mask] = next_token

        return idx


    def generate(self, idx: torch.Tensor, temperature: float = 0.0, 
               kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
               levels: Optional[torch.Tensor] = None): 
        
        is_cached_pass = kv_cache is not None
        
        if is_cached_pass:
            assert idx.shape[1] == 1, "For cached generation, provide only the new token"
            assert levels is not None, "Levels must be provided for cached generation"
            
            x = self.transformer.wte(idx)
            
            q_idx_val = levels.shape[1]
            
            def causal_mask(b, h, q_idx, kv_idx):
                current_q_pos = q_idx_val + q_idx
                
                is_higher_level = levels[b, kv_idx] > 0
                is_recent = (current_q_pos - kv_idx) <= self.memory_span
                keep_mask = is_higher_level | is_recent
                return keep_mask
            
            q_len = 1
            kv_len = q_idx_val + 1
            
            new_level = infer_level(idx, self.vocab_sizes, self.level_mask_tokens[0])
            levels = torch.cat([levels, new_level], dim=1)
        else:
            x = self.transformer.wte(idx) 
            levels = infer_level(idx, self.vocab_sizes, self.level_mask_tokens[0])

            def causal_mask(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                
                is_higher_level = levels[b, kv_idx] > 0
                is_recent = (q_idx - kv_idx) <= self.memory_span
                keep_mask = is_higher_level | is_recent 
                return causal_mask & keep_mask
                
            S = idx.shape[1]
            q_len = S
            kv_len = S
            kv_cache = [None] * self.num_layers

        block_mask = create_block_mask(causal_mask, None, None, q_len, kv_len, device=self.device, _compile=self._compile)
        
        x = norm(x)
        x0 = x
        v1 = None
        
        new_kv_cache = []
        for i in range(self.num_layers):
            layer_cache = kv_cache[i] if is_cached_pass else None
            x, v1, updated_cache = self.transformer.h[i](
                x, v1, x0, block_mask, 
                cache=layer_cache, 
                cache_offset=0 if not is_cached_pass else kv_cache[0][0].shape[1]
            )
            new_kv_cache.append(updated_cache)

        x = norm(x)

        next_token = self._decode(self.lm_head(x[:, -1, :]), temperature=temperature)

        return next_token, new_kv_cache, levels


    def _decode(self, logits: torch.Tensor, levels: Optional[torch.Tensor] = None, temperature: float = 0.0):
                
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        logits[:, self.level_mask_tokens] = float('-inf') # Invalidate ALL MASK_TOK logits

        if levels is not None:
            assert levels.shape == logits.shape[:-1], "Levels and logits must have the same shape except for the last dimension"
            start_logits = self.level_vocab_starts[levels]
            end_logits = self.level_vocab_ends[levels]

            vocab_indices = torch.arange(logits.size(-1), device=self.device)
            mask = (vocab_indices >= start_logits.unsqueeze(-1)) & (vocab_indices < end_logits.unsqueeze(-1))

            logits = torch.where(mask, logits, torch.tensor(-float('inf')))

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
        return next_token