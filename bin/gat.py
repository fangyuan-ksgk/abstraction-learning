from model import Block, CastedLinear, create_block_mask
from utils import create_loss_mask, HierSeq
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch
import itertools
from constant import MASK_TOK
import torch.nn.functional as F
from model import norm
import torch.nn as nn
from typing import Union
from pathlib import Path

# util function
# ------------------------------------------------------------------------
def infer_level(indices: torch.Tensor, vocab_sizes: torch.Tensor, pad_token: int):
    indices_expanded = indices.unsqueeze(-1)  # [batch_size, seq_len, 1]
    levels = (indices_expanded < vocab_sizes.cumsum(dim=0)).int().argmax(dim=-1)

    padding_mask = (indices == pad_token)
    final_levels = torch.where(padding_mask, -1, levels.long())
    return final_levels

def infer_timestamp(levels: torch.Tensor, K: int, l: int = 1) -> torch.Tensor:
    is_level = (levels == l-1).long()  
    cumulative_counts = torch.cumsum(is_level, dim=-1)
    timestamps = (cumulative_counts - 1) // K
    timestamps.clamp_(min=0) # this assings the correct timestamp 
    return levels, timestamps
# ------------------------------------------------------------------------s

@dataclass
class GATConfig:
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    flex_kernel_options: Optional[dict] = None
    K: int = 4  # abstraction ratio
    L: int = 4  # total # of levels (including 0-th level)
    vocab_size_list: list = field(default_factory=lambda: [128, 64, 32])
    device: str = "cuda"
    _compile: bool = True


# New version, aligned with GPT architecture, without level-embedding
class reGAT(nn.Module): 

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.n_layer
        self.L = config.L
        self.K = config.K

        # multi-level vocab specific parameters
        self.vocab_sizes = torch.tensor(config.vocab_size_list, device=config.device) + 1
        self.total_vocab_size = sum(self.vocab_sizes)
        self.level_mask_tokens = self.vocab_sizes.cumsum(dim=0) - 1
        self.level_vocab_starts = torch.concat([torch.tensor([0.], device=config.device), self.vocab_sizes])[:-1]
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
            x, v1, _ = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
        loss = loss.view(idx.shape[0], idx.shape[1])
        loss[torch.isin(target, self.level_mask_tokens)] = 0 # ignore loss for mask tokens
        return loss


    def denoise(self, idx: torch.Tensor, denoise_mask: torch.Tensor, temperature: float = 0.0): 

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
            x, v1, _ = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        levels = infer_level(idx, self.vocab_sizes, self.level_mask_tokens[0])
        next_token = self._decode(self.lm_head(x[denoise_mask]), levels=levels[denoise_mask], temperature=temperature)
        idx[denoise_mask] = next_token

        return idx


    def generate(self, idx: torch.Tensor, temperature: float = 0.0, kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None): 
        
        is_cached_pass = kv_cache is not None
        
        if is_cached_pass:
            assert idx.shape[1] == 1, "For cached generation, provide only the new token"
            x = self.transformer.wte(idx)  # [batch_size, 1, n_embd]
            cache_len = kv_cache[0][0].shape[1] if kv_cache[0] is not None else 0
            total_len = cache_len + 1
            def causal_mask(b, h, q_idx, kv_idx):
                return kv_idx >= 0  # Always true since we're at the end of sequence
            q_len = 1
            kv_len = total_len
        else:
            x = self.transformer.wte(idx)  # [batch_size, seq_len, n_embd]
            def causal_mask(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                return causal_mask
            S = idx.shape[1]
            q_len = S
            kv_len = S
            kv_cache = [None] * self.num_layers  # Initialize cache list

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

        return next_token, new_kv_cache


    def _decode(self, logits: torch.Tensor, levels: Optional[torch.Tensor] = None, temperature: float = 0.0):
                
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        logits[:, self.level_vocab_ends-1] = float('-inf') # Invalidate ALL MASK_TOK logits

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



class GAT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.n_layer
        self.L = config.L
        self.K = config.K

        self.original_vocab_sizes = list(config.vocab_size_list)
        self.level_mask_tokens = torch.tensor(self.original_vocab_sizes)
        new_vocab_size_list = [size + 1 for size in self.original_vocab_sizes] # per-level mask token addition

        self.vocab_size_list = new_vocab_size_list
        vocab_offsets = torch.tensor([0] + list(itertools.accumulate(self.vocab_size_list)))[:-1]
        self.register_buffer('vocab_offsets', vocab_offsets)
        self.total_vocab_size = sum(self.vocab_size_list)

        self.wtes = nn.Embedding(self.total_vocab_size, config.n_embd)
        self.lm_heads = CastedLinear(config.n_embd, self.total_vocab_size)
        
        self.lm_heads.weight.data.zero_()
        
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.level_embeddings = nn.Parameter(torch.randn(config.L, config.n_embd))

        self.device = config.device
        self._compile = config._compile

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
            x, v1, _ = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        return self._compute_ppt(x, batch_data)
    
    def generate(self, batch_data: HierSeq, parallel: bool = False, temperature: float = 0.0,  
                 levels: Optional[torch.Tensor] = None):

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
            x, v1, _ = self.transformer.h[i](x, v1, x0, block_mask)

        x = norm(x)

        if parallel:
            batch_data = self._denoise(x, batch_data, temperature)
        else:
            batch_data = self._generate(x, batch_data, temperature, levels)

        return batch_data

    def save_checkpoint(self, path: Union[str, Path]):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: Union[str, Path], strict: bool = True):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=strict)

    def _create_hseq_embd(self, batch_data: HierSeq, do_slice: bool = True) -> torch.Tensor: 
        if do_slice: 
            input_idx, input_levels = batch_data.tokens[:-1], batch_data.levels[:-1]
        else: 
            input_idx, input_levels = batch_data.tokens, batch_data.levels

        offset_indices = input_idx + self.vocab_offsets[input_levels]
        
        x = self.wtes(offset_indices).unsqueeze(0) + self.level_embeddings[input_levels].unsqueeze(0)

        return x

    def _create_hseq_embd_for_token(self, token: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        offset_index = token + self.vocab_offsets[level]
        x = self.wtes(offset_index).unsqueeze(0).unsqueeze(0) + self.level_embeddings[level].unsqueeze(0).unsqueeze(0)
        return x

    def _compute_ppt(self, x: torch.Tensor, batch_data: HierSeq) -> torch.Tensor: 
        target_idx, target_levels = batch_data.tokens[1:], batch_data.levels[1:]
        loss_mask = create_loss_mask(batch_data.sample_idx)[1:]

        all_logits = self.lm_heads(x)
        all_logits = 30 * torch.tanh(all_logits / 30).float()

        masked_logits = all_logits[:, loss_mask, :]
        masked_target_idx = target_idx[loss_mask]
        masked_target_levels = target_levels[loss_mask]

        offset_target_idx = masked_target_idx + self.vocab_offsets[masked_target_levels]

        all_losses = F.cross_entropy(
            masked_logits.view(-1, self.total_vocab_size),
            offset_target_idx,
            reduction="none"
        )

        ppt = torch.zeros_like(target_idx, device=self.device, dtype=torch.float32)
        ppt[loss_mask] = all_losses

        return ppt

    
    def _decode(self, logits: torch.Tensor, levels: Optional[torch.Tensor] = None, temperature: float = 0.0):
        """Decode logits into tokens and levels"""

        logits = 30 * torch.tanh(logits / 30)

        mask_token_global_indices = self.vocab_offsets + self.level_mask_tokens.to(self.device)
        logits[:, mask_token_global_indices] = float('-inf') # Invalidate ALL MASK_TOK logits

        if levels is not None:
            assert levels.shape == logits.shape[:-1], "Levels and logits must have the same shape except for the last dimension"
            
            start_offsets = self.vocab_offsets[levels]
            end_offsets = start_offsets + torch.tensor(self.vocab_size_list, device=self.device)[levels]
            vocab_range = torch.arange(self.total_vocab_size, device=self.device)
            mask = (vocab_range >= start_offsets.unsqueeze(1)) & (vocab_range < end_offsets.unsqueeze(1))

            masked_logits = torch.where(mask, logits, torch.tensor(float('-inf'), device=self.device))  

            if temperature == 0.0:
                next_tokens_flat = torch.argmax(masked_logits, dim=-1)
            else:
                next_tokens_flat = torch.multinomial(F.softmax(masked_logits / temperature, dim=-1), num_samples=1).squeeze(-1)
            
            next_levels = levels

        else:
            if temperature == 0.0:
                next_tokens_flat = torch.argmax(logits, dim=-1)
            else:
                next_tokens_flat = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
            
            next_levels = (next_tokens_flat.unsqueeze(1) >= self.vocab_offsets.unsqueeze(0)).sum(dim=1) - 1

        level_specific_tokens = next_tokens_flat - self.vocab_offsets[next_levels]
        
        return level_specific_tokens, next_levels


    def _generate(self, x: torch.Tensor, batch_data: HierSeq, temperature: float = 0.0, levels: Optional[torch.Tensor] = None):
        """
        Performs a single auto-regressive generation step.
        The model decides the level of the new token for each sample.
        """

        with torch.no_grad():

            # When generating with a cache, x is the hidden state for the last token of each sample
            if x.shape[1] == 1:
                last_hidden_states = x.squeeze(0)
            else:
                _, last_indices = torch.unique_consecutive(batch_data.sample_idx, return_inverse=True)
                last_token_positions = torch.where(last_indices[:-1] != last_indices[1:])[0]
                last_token_positions = torch.cat([last_token_positions, torch.tensor([len(batch_data.sample_idx)-1], device=self.device)])
                last_hidden_states = x[0, last_token_positions, :]

            logits = self.lm_heads(last_hidden_states)
            new_tokens, new_levels = self._decode(logits, levels=levels, temperature=temperature)
            
            batch_data.append_tokens(new_tokens, new_levels)

        return batch_data

    def _denoise(self, x: torch.Tensor, batch_data: HierSeq, temperature: float = 0.0):
        """
        Fills in all MASK_TOK placeholders in a single parallel pass, using the hidden
        state from the token *before* the mask for prediction. This simulates a
        parallel auto-regressive step.
        """
        
        is_mask = (batch_data.tokens == self.level_mask_tokens.to(self.device)[batch_data.levels])
        mask_positions = is_mask.nonzero(as_tuple=True)[0]
        
        if mask_positions.numel() == 0:
            return batch_data

        # Parallel AR generation | differs slighly to denoising
        prediction_positions = mask_positions - 1
        prediction_positions = torch.clamp(prediction_positions, min=0)
        
        hidden_states = x[0, prediction_positions, :]

        levels_to_decode = batch_data.levels[mask_positions]
        
        logits = self.lm_heads(hidden_states)
        
        new_tokens, _ = self._decode(logits, levels=levels_to_decode, temperature=temperature)
        
        batch_data.tokens = batch_data.tokens.scatter(0, mask_positions, new_tokens)
        
        return batch_data