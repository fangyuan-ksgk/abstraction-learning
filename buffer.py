# Buffer Management & Data Save/Load
# ------------------------------------------------------------ 
import numpy as np 
import struct 
import os
from utils import HierSeq 
import torch 

def write_shard(filename, samples, timestamps):
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = 20241220, 1, len(samples)
    
    total_bytes = 0
    for (sample_len, hier_seq), timestamp in zip(samples, timestamps):
        total_bytes += 5 + sum(4 + len(tokens) * 4 for tokens in hier_seq)
        total_bytes += sum(4 + len(ts) * 4 for ts in timestamp)  # Add bytes for timestamp hierarchy
    header[3] = total_bytes
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        for (sample_len, hier_seq), timestamp in zip(samples, timestamps):
            f.write(struct.pack('IB', sample_len, len(hier_seq)))
            for tokens in hier_seq:
                f.write(struct.pack('I', len(tokens)))
                if tokens:
                    f.write(np.array(tokens, dtype=np.int32).tobytes())
            for ts in timestamp:
                f.write(struct.pack('I', len(ts)))
                if ts:
                    f.write(np.array(ts, dtype=np.int32).tobytes())

def load_shard(filename):
    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(1024), dtype=np.int32)
        mm = np.memmap(filename, dtype='uint8', mode='r', offset=1024)
        
        samples, timestamps, offset = [], [], 0
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
            
            timestamp = []
            for _ in range(n_levels):
                n_ts = struct.unpack_from('I', mm, offset)[0]
                offset += 4
                ts = np.frombuffer(mm, dtype=np.int32, count=n_ts, offset=offset).tolist() if n_ts else []
                offset += n_ts * 4
                timestamp.append(ts)
            
            samples.append((sample_len, hier_seq))
            timestamps.append(timestamp)
    return samples, timestamps


class Buffer: 
    
    def __init__(self, file_path, max_length, K, L, ppl_thres_percentile=0.2): 
        self.file_path = file_path
        self.pool, self.timestamps = load_shard(file_path)  # Now load_shard returns both
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
            timestamp = self.timestamps[idx]
            if curr_len + sample_len > self.max_length:
                break
            batch.append((hier_seq, timestamp))
            curr_len += sample_len
            selected_indices.append(idx)
        
        return HierSeq.from_hierarchical_data(batch, self.K, self.L, sample_indices=selected_indices)

    def update(self, hseq: HierSeq, cr: torch.Tensor, ppl: torch.Tensor): 

        indices = torch.unique(hseq.sample_idx, sorted=True)

        for idx, cr_val, ppl_val in zip(indices, cr, ppl): 
            sample_len = self.pool[idx][0]
            sample_hier_seq, sample_timestamp = hseq.to_hierarchical_data()[idx]
            self.pool[idx] = (sample_len, sample_hier_seq)
            self.timestamps[idx] = sample_timestamp
            self.cr[idx] = cr_val
            self.ppl[idx] = ppl_val

        write_shard(self.file_path, self.pool, self.timestamps)  # Pass timestamps too
