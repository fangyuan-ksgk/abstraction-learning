# Buffer Management & Data Save/Load
# ------------------------------------------------------------ 
import numpy as np 
import struct 

def save_sequences_and_lengths(sequences, lengths, filepath):
    """Save sequences and lengths to a binary file."""
    with open(filepath, 'wb') as f:
        # Write number of sequences
        f.write(struct.pack('I', len(sequences)))
        
        for seq, length in zip(sequences, lengths):
            # Write length first
            f.write(struct.pack('I', length))
            # Write sequence length and data
            f.write(struct.pack('I', len(seq)))
            f.write(np.array(seq, dtype=np.int32).tobytes())
    
    print(f"Saved {len(sequences)} sequences to {filepath}")

def load_sequences_and_lengths(filepath):
    """Load sequences and lengths from a binary file."""
    sequences = []
    lengths = []
    
    with open(filepath, 'rb') as f:
        n_sequences = struct.unpack('I', f.read(4))[0]
        
        for _ in range(n_sequences):
            # Read length
            length = struct.unpack('I', f.read(4))[0]
            lengths.append(length)
            
            # Read sequence
            seq_len = struct.unpack('I', f.read(4))[0]
            seq = np.frombuffer(f.read(seq_len * 4), dtype=np.int32).tolist()
            sequences.append(seq)
    
    print(f"Loaded {len(sequences)} sequences from {filepath}")
    return sequences, lengths

