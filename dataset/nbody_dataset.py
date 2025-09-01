from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import json
import struct
import numpy as np
from dataset.nbody import create_dataset_with_params, TinyTokenizer
from utils import compute_hier_seq_len

@dataclass
class NBodyDataset:
    """N-body dataset with save/load functionality"""
    # Core parameters
    n_bodies: int = 2
    patterns: List[str] = field(default_factory=lambda: ['cartesian'])
    n_context: int = 3
    stride: int = 1
    T: int = 10
    include_masses: bool = True
    K: int = 2
    L: int = 3
    filepath: str = 'dataset/nbody/sequences.bin'
    n_samples: Optional[int] = None # max number of samples to build

    # Data storage
    sequences: List = field(default_factory=list, init=False)
    lengths: List = field(default_factory=list, init=False)
    
    def build(self):
        """Generate and save the dataset"""
        # Generate raw data
        dataset = create_dataset_with_params(
            n_bodies=self.n_bodies,
            patterns=self.patterns,
            n_context=self.n_context,
            stride=self.stride,
            T=self.T,
            include_masses=self.include_masses,
        )
        
        # Tokenize sequences
        tokenizer = TinyTokenizer()
        self.sequences = [tokenizer(s) for s in dataset['sequences']]
        self.lengths = [compute_hier_seq_len(seq, self.L, self.K) for seq in self.sequences]

        if self.n_samples is not None:
            self.sequences = self.sequences[:self.n_samples]
            self.lengths = self.lengths[:self.n_samples]
        
        # Save to disk
        self._save()
        return self
    
    def load(self):
        """Load dataset from disk"""
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
            json.dump({
                'n_bodies': self.n_bodies,
                'patterns': self.patterns,
                'n_context': self.n_context,
                'stride': self.stride,
                'T': self.T,
                'include_masses': self.include_masses,
                'K': self.K,
                'L': self.L,
            }, f, indent=2)
        
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
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx] 

    def update_abstract_params(self, L: Optional[int] = None, K: Optional[int] = None): 
        self.L = L if L is not None else self.L
        self.K = K if K is not None else self.K
        self.lengths = [compute_hier_seq_len(seq, self.L, self.K) for seq in self.sequences]


# Example Dataset Creation
# -------------------------------------------------------------

# # Create and build new dataset
# dataset = NBodyDataset(
#     n_bodies=2,
#     patterns=['circular'],
#     T=10,
#     filepath='dataset/nbody/2body_100.bin',
#     n_samples=100
# ).build()