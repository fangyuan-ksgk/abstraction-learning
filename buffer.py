# Buffer Management & Data Save/Load
# ------------------------------------------------------------ 
import numpy as np 
import struct 
import os

def write_buffer_shard(filename, samples):
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


def load_buffer_shard(filename):
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


def update_sample_in_shard(filename, sample_idx, new_hier_seq):
    """
    Shard-level rewrites (Room for optimization)
    """
    samples = load_buffer_shard(filename)
    
    old_sample_len, _ = samples[sample_idx]
    samples[sample_idx] = (old_sample_len, new_hier_seq)
    
    write_buffer_shard(filename, samples)



# class ShardedBufferManager:
#     """Manage multiple buffer shards with updates."""
    
#     def __init__(self, shard_pattern='buffer_*.bin', cache_size=2):
#         self.files = sorted(glob.glob(shard_pattern))
#         if not self.files:
#             raise ValueError(f"No files found matching {shard_pattern}")
        
#         self.cache = {}  # filename -> (header, samples)
#         self.cache_size = cache_size
#         self.file_iter = itertools.cycle(self.files)
        
#     def get_batch(self, max_length, L, K):
#         """Get batch from shards with cycling."""
#         while True:
#             filename = next(self.file_iter)
            
#             # Load with caching
#             if filename not in self.cache:
#                 if len(self.cache) >= self.cache_size:
#                     # Evict oldest
#                     self.cache.pop(next(iter(self.cache)))
#                 self.cache[filename] = load_buffer_shard(filename)
            
#             header, samples = self.cache[filename]
            
#             # Build batch
#             batch = []
#             curr_len = 0
#             indices = np.random.permutation(len(samples))
            
#             for idx in indices:
#                 sample_len, hier_seq = samples[idx]
#                 if curr_len + sample_len > max_length:
#                     break
#                 batch.append((hier_seq, None))
#                 curr_len += sample_len
            
#             if batch:
#                 return HierSeq.from_hierarchical_data(batch, K, L), filename, indices[:len(batch)]
    
#     def update_samples(self, filename, indices, batch_data):
#         """Update abstract tokens for samples in batch."""
#         if filename in self.cache:
#             _, samples = self.cache[filename]
            
#             for i, idx in enumerate(indices):
#                 sample_mask = batch_data.sample_idx == i
#                 sample_len, old_hier = samples[idx]
                
#                 # Extract updated hierarchy
#                 new_hier = [old_hier[0]]  # Keep level 0
#                 for l in range(1, batch_data.L):
#                     level_mask = sample_mask & (batch_data.levels == l)
#                     if level_mask.any():
#                         new_hier.append(batch_data.tokens[level_mask].cpu().numpy().tolist())
#                     else:
#                         new_hier.append([])
                
#                 samples[idx] = (sample_len, new_hier)
            
#             # Write back to disk
#             write_buffer_shard(filename, samples)
#             # Update cache
#             self.cache[filename] = (self.cache[filename][0], samples)
