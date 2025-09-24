from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


# Basedataset 
# -----------------------------------------------------------------------
@dataclass
class BaseDataset(ABC): 
    filepath: str 
    num_data: Optional[int] = None
    vocab_size_list: Optional[List[int]] = None

    sequences: list = field(default_factory=list, init=False)

    @abstractmethod
    def build(self): 
        raise NotImplementedError("Subclasses must implement build method")

    def load(self): 
        self.sequences = self._load_sequences()
        return self

    def _save(self): 
        """Save sequences and lengths to binary file"""
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.filepath, 'wb') as f:
            # Write header
            header = np.array([20241220, 1, len(self.sequences)], dtype=np.int32)
            f.write(header.tobytes())
            
            # Write sequences and lengths
            for seq in self.sequences:
                f.write(np.int32(len(seq)).tobytes())
                f.write(np.array(seq, dtype=np.int32).tobytes())
        
        # Save config
        config_path = self.filepath.replace('.bin', '_config.json')
        with open(config_path, 'w') as f:
            config = {}
            excluded_fields = {'sequences', 'tokenizer'}
            
            for field_name in self.__dataclass_fields__:
                if field_name not in excluded_fields:
                    config[field_name] = getattr(self, field_name)
            
            json.dump(config, f, indent=2)
        
        print(f"Saved {len(self.sequences)} sequences to {self.filepath}")

    def _load_sequences(self) -> Tuple[List, List]:
        """Load sequences and lengths from binary file"""
        sequences = []
        
        with open(self.filepath, 'rb') as f:
            # Read header
            header = np.frombuffer(f.read(12), dtype=np.int32)
            n_sequences = header[2]
            
            # Read sequences
            for _ in range(n_sequences):
                # FIX: Read only one length value, which is the sequence length
                seq_len = np.frombuffer(f.read(4), dtype=np.int32)[0]
                seq = np.frombuffer(f.read(seq_len * 4), dtype=np.int32).tolist()
                
                sequences.append(seq)
        
        return sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx] 


    @classmethod
    def from_file(cls, filepath: str):
        """Load dataset from existing file"""
        # Try to load config if it exists
        config_path = filepath.replace('.bin', '_config.json')
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # The filepath argument to this method takes precedence over the one in the config.
            if 'filepath' in config:
                del config['filepath']
            
            dataset = cls(**config, filepath=filepath)
        else:
            dataset = cls(filepath=filepath)
        
        return dataset.load()


def get_batch(dataset: BaseDataset, batch_size: int, max_length: int, pad_token_id: int, device: str = "cpu"):
    indices = np.random.randint(0, len(dataset), size=batch_size)
    batch_sequences = [torch.tensor(dataset[i][:max_length], dtype=torch.long) for i in indices]

    batch = pad_sequence(batch_sequences, batch_first=True, padding_value=pad_token_id)
    return batch.to(device)