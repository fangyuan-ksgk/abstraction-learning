# Buffer Management & Data Save/Load
# ------------------------------------------------------------ 
import numpy as np 
import struct 
import os
from utils import HierSeq 
import torch 

# (TBD). No timestamp is saved yet. This requires change when sparse memory is included. 
def write_shard(filename, samples):
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = 20241220, 1, len(samples)
    
    total_bytes = 0
    for sample_len, hier_seq in samples:
        total_bytes += 5 + sum(4 + len(tokens) * 4 for tokens in hier_seq)
    header[3] = total_bytes
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        for sample_len, hier_seq in samples:
            f.write(struct.pack('IB', sample_len, len(hier_seq)))
            for tokens in hier_seq:
                f.write(struct.pack('I', len(tokens)))
                if tokens:
                    f.write(np.array(tokens, dtype=np.int32).tobytes())

# (TBD). No timestamp is loaded yet. This requires change when sparse memory is included. 
def load_shard(filename):
    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=np.int32)
        mm = np.memmap(filename, dtype='uint8', mode='r', offset=1024)
        
        samples, offset = [], 0
        for _ in range(header[2]):
            sample_len, n_levels = struct.unpack_from('IB', mm, offset)
            offset += 5
            
            hier_seq = []
            for _ in range(n_levels):
                n_tokens = struct.unpack_from('I', mm, offset)[0]
                offset += 4
                tokens = np.frombuffer(mm, dtype=np.int32, count=n_tokens, offset=offset).tolist() if n_tokens else []
                offset += n_tokens * 4
                hier_seq.append(tokens)
            
            samples.append((sample_len, hier_seq))  # This line is correct
    return samples


def update_shard(filename, sample_idx, new_hier_seq):
    """
    Shard-level rewrites (Room for optimization)
    """
    samples = load_shard(filename)
    
    old_sample_len, _ = samples[sample_idx]
    samples[sample_idx] = (old_sample_len, new_hier_seq)
    
    write_shard(filename, samples)


# (TBD). No timestamp is maintained yet. This requires change when sparse memory is included. 
class Buffer: 
    
    def __init__(self, file_path, max_length, K, L, ppl_thres_percentile=0.2): 
        self.pool = load_shard(file_path)
        self.max_length = max_length
        self.cr = np.array([0. for _ in range(len(self.pool))]) # control rate per sample
        self.ppl = np.array([0. for _ in range(len(self.pool))]) # perplexity per sample
        self.ppl_thres_percentile = ppl_thres_percentile
        self.K, self.L = K, L

    @property
    def ppl_thres(self): 
        return np.percentile(self.ppl, self.ppl_thres_percentile)

    def get_batch(self):

        sorted_indices = sorted(range(len(self.pool)), key=lambda i: (self.cr[i], self.ppl[i]))

        batch = []
        curr_len = 0
        selected_indices = []
        
        for idx in sorted_indices:
            sample_len, hier_seq = self.pool[idx]
            if curr_len + sample_len > self.max_length:
                break
            batch.append((hier_seq, None))
            curr_len += sample_len
            selected_indices.append(idx)
        
        return HierSeq.from_hierarchical_data(batch, self.K, self.L, sample_indices=selected_indices)

    def update(self, hseq: HierSeq, cr: torch.Tensor, ppl: torch.Tensor): 

        indices = torch.unique(hseq.sample_idx, sorted=True)

        for idx, cr_val, ppl_val in zip(indices, cr, ppl): 
            # (TBD) .to_hierarchical_data() is not implemented yet
            self.pool[idx] = (self.pool[idx][0], hseq.to_hierarchical_data()[idx])
            self.cr[idx] = cr_val
            self.ppl[idx] = ppl_val

        write_shard(self.file_path, self.pool)
