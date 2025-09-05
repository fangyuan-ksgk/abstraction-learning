from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import json
import struct
import numpy as np
from nbody import create_dataset_with_params, TinyTokenizer
from base import compute_hier_seq_len
from base import BaseDataset 

@dataclass 
class NBodyDataset(BaseDataset): 
    n_bodies: int = 2
    patterns: List[str] = field(default_factory=lambda: ['cartesian'])
    n_context: int = 3
    stride: int = 1
    T: int = 10
    include_masses: bool = True
    K: int = 2
    L: int = 3
    filepath: str = 'dataset/nbody/sequences.bin'
    tokenizer: TinyTokenizer = TinyTokenizer()
    answer_token: str = TinyTokenizer().answer_token
    answer_token_id: int = TinyTokenizer().answer_token_id

    def build(self): 
        # Generate raw data
        dataset = create_dataset_with_params(
            n_bodies=self.n_bodies,
            patterns=self.patterns,
            n_context=self.n_context,
            stride=self.stride,
            T=self.T,
            include_masses=self.include_masses,
            answer_token=self.answer_token,
        )
        
        # Tokenize sequences
        self.vocab_size = self.tokenizer.vocab_size
        self.sequences = [self.tokenizer(s) for s in dataset['sequences']]
        self.lengths = [compute_hier_seq_len(seq, self.L, self.K) for seq in self.sequences]

        if self.num_data is not None:
            self.sequences = self.sequences[:self.num_data]
            self.lengths = self.lengths[:self.num_data]
        
        # Save to disk
        self._save()
        return self


# Example Dataset Creation
# -------------------------------------------------------------

# # Create and build new dataset
# dataset = NBodyDataset(
#     n_bodies=2,
#     patterns=['circular'],
#     T=10,
#     filepath='dataset/nbody/2body_100.bin',
#     num_data=100
# ).build()