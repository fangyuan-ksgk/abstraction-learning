from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import json


# util function
# -----------------------------------------------------------------------
def compute_hier_seq_len(seq: list, L: int, K: int) -> int:
    """Full length with abstraction computation based on trajectory sequence alone"""
    seq_len = len(seq)
    total = seq_len
    for l in range(1, L):
        total += (seq_len - 1) // (K ** l) + 1
    return total + 1

# Basedataset 
# -----------------------------------------------------------------------
@dataclass
class BaseDataset(ABC): 
    num_data: int
    filepath: str 
    vocab_size: Optional[int] = None 
    L: int = 1
    K: int = 8

    sequences: list = field(default_factory=list, init=False)
    lengths: list = field(default_factory=list, init=False)

    tokenizer: Optional[Callable] = None

    @abstractmethod
    def build(self): 
        raise NotImplementedError("Subclasses must implement build method")

    def load(self): 
        self.sequences, self.lengths = self._load_sequences_and_lengths()
        return self

    def _save(self): 
        """Save sequences and lengths to binary file"""
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.filepath, 'wb') as f:
            # Write header
            header = np.array([20241220, 1, len(self.sequences)], dtype=np.int32)
            f.write(header.tobytes())
            
            # Write sequences and lengths
            for seq, length in zip(self.sequences, self.lengths):
                f.write(np.int32(length).tobytes())
                f.write(np.int32(len(seq)).tobytes())
                f.write(np.array(seq, dtype=np.int32).tobytes())
        
        # Save config
        config_path = self.filepath.replace('.bin', '_config.json')
        with open(config_path, 'w') as f:
            config = {}
            excluded_fields = {'sequences', 'lengths', 'tokenizer'}
            
            for field_name in self.__dataclass_fields__:
                if field_name not in excluded_fields:
                    config[field_name] = getattr(self, field_name)
            
            config['token_count'] = len(self.sequences)
            json.dump(config, f, indent=2)
        
        print(f"Saved {len(self.sequences)} sequences to {self.filepath}")

    def _load_sequences_and_lengths(self) -> Tuple[List, List]:
        """Load sequences and lengths from binary file"""
        sequences, lengths = [], []
        
        with open(self.filepath, 'rb') as f:
            # Read header
            header = np.frombuffer(f.read(12), dtype=np.int32)
            n_sequences = header[2]
            
            # Read sequences
            for _ in range(n_sequences):
                length = np.frombuffer(f.read(4), dtype=np.int32)[0]
                seq_len = np.frombuffer(f.read(4), dtype=np.int32)[0]
                seq = np.frombuffer(f.read(seq_len * 4), dtype=np.int32).tolist()
                
                lengths.append(int(length))
                sequences.append(seq)
        
        return sequences, lengths

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx] 

    @classmethod
    def from_file(cls, filepath: str):
        """Load dataset from existing file"""
        # Try to load config if it exists
        config_path = filepath.replace('.bin', '_config.json')
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            dataset = cls(**config, filepath=filepath)
        else:
            dataset = cls(filepath=filepath)
        
        return dataset.load()

    def update_abstract_params(self, L: Optional[int] = None, K: Optional[int] = None): 
        self.L = L if L is not None else self.L
        self.K = K if K is not None else self.K
        self.lengths = [compute_hier_seq_len(seq, self.L, self.K) for seq in self.sequences]
