# Buffer Management & Data Save/Load
# ------------------------------------------------------------ 
import numpy as np 
import struct 
import os
from utils import HierSeq, pad_abstract_tokens
import torch 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns

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


def plot_buffer_state(ax, cr, ppl, ppl_thres, ppl_thres_percentile, 
                      xlim=None, ylim=None, title=None, stats_text=None, show_colorbar=True):
    """Utility function to plot buffer state on a given axis"""
    sorted_indices = sorted(range(len(cr)), key=lambda i: (cr[i], ppl[i]))
    priority_colors = np.array([sorted_indices.index(i) for i in range(len(cr))])
    
    scatter = ax.scatter(cr, ppl, c=priority_colors, 
                        cmap='viridis_r', s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                        vmin=0, vmax=len(cr)-1)
    
    ax.set_xlabel('Control Rate', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.grid(True, alpha=0.3)
    ax.axhline(y=ppl_thres, color='red', linestyle='--', alpha=0.5, 
              label=f'PPL threshold (p{ppl_thres_percentile}): {ppl_thres:.2f}')
    
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Selection Priority', fontsize=11)
        # Add custom tick labels
        cbar.set_ticks([0, len(cr)//4, len(cr)//2, 3*len(cr)//4, len(cr)-1])
        cbar.set_ticklabels(['Highest\n(Selected First)', 'High', 'Medium', 'Low', 'Lowest\n(Selected Last)'])
    
    if stats_text:
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    return scatter

class Buffer: 
    
    def __init__(self, file_path, max_length, K, L, ppl_thres_percentile=20.0, t_record = 10, debug=False): 
        self.file_path = file_path
        self.pool, self.timestamps = load_shard(file_path)  # Now load_shard returns both
        if debug: 
            self.pool = self.pool[:120]
            self.timestamps = self.timestamps[:120]
        self.max_length = max_length

        self.t_record = t_record
        self.cr = np.zeros((t_record, len(self.pool)))
        self.ppl = np.full((t_record, len(self.pool)), float('inf'))
        self.cts = np.zeros((t_record, len(self.pool)), dtype=int)

        self.ppl_thres_percentile = ppl_thres_percentile
        self.K, self.L = K, L

    @property
    def ppl_thres(self): 
        # Filter out infinite values when calculating percentile
        finite_ppl = self.ppl[-1, np.isfinite(self.ppl[-1])]
        if len(finite_ppl) == 0:
            return float('inf')  # If no finite values, return inf
        return np.percentile(finite_ppl, self.ppl_thres_percentile)

    def get_batch(self, pad: bool = True, t_search: int = 2, noise_scale: float = 0.1):

        # perturbate on index selection
        cr_range = np.max(self.cr[-1]) - np.min(self.cr[-1]) + 1e-8
        finite_ppl = self.ppl[-1, np.isfinite(self.ppl[-1])]
        ppl_range = np.max(finite_ppl) - np.min(finite_ppl) + 1e-8 if len(finite_ppl) > 0 else 1.0
        cr_noise = np.random.randn(len(self.pool)) * cr_range * noise_scale
        ppl_noise = np.random.randn(len(self.pool)) * ppl_range * noise_scale
        
        sorted_indices = sorted(range(len(self.pool)), 
                              key=lambda i: (self.cr[-1, i] + cr_noise[i], 
                                           self.ppl[-1, i] + ppl_noise[i]))
        
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

        batch_data = HierSeq.from_hierarchical_data(batch, self.K, self.L, sample_indices=selected_indices)
        if pad: 
            cts = self.cts[-1, selected_indices]
            pad_abstract_tokens(batch_data, cts, t_search)

        return batch_data

    def update(self, hseq: HierSeq, cr: torch.Tensor, ppl: torch.Tensor, cts: torch.Tensor): 

        indices = torch.unique(hseq.sample_idx, sorted=True)

        batch_seqs, batch_ts = hseq.to_hierarchical_data()

        for loc_idx, global_idx in enumerate(indices): 
            sample_len = self.pool[global_idx][0]
            sample_hier_seq, sample_ts = batch_seqs[loc_idx], batch_ts[loc_idx]
            self.pool[global_idx] = (sample_len, sample_hier_seq)
            self.timestamps[global_idx] = sample_ts
            self._update_record(cr[loc_idx], ppl[loc_idx], cts[loc_idx])

    # (TBD). 'ppl improvement with search' & 'ppl improvement w.o. search' should be computed
    #         we'd use percentile of ppl improvement to determine whehter to do search or optimization, dynamic threshold
    #        'with search' means cts[-1, idx] >= 0, 'without search' means cts[-1, idx] == -1
    def _update_record(self, cr, ppl, cts): 
        self.cr = np.roll(self.cr, -1, axis=0)
        self.ppl = np.roll(self.ppl, -1, axis=0)
        self.cts = np.roll(self.cts, -1, axis=0)
        self.cr[-1, :] = cr
        self.ppl[-1, :] = ppl
        self.cts[-1, :] = cts

    # (TBD). An issue, or rather the exact issue, is about 'stop repeativive search ON SAME POSITION' rather than stop search entirely
    #        In a sense, the indication of 'no improvement from search' should not lead to stop search (cts=-1), rather, it should lead
    #        to 'no backtracking', or at the very least, it should lead to 'no backtracking on the same old position', this requires 
    #        some gadget & function to implement

    @property 
    def search_mask(self): 
        valid_ppl = ~np.isinf(self.ppl)
        valid_pairs = valid_ppl[1:] & valid_ppl[:-1]
        valid_search = (self.cts[1:] != -1) & valid_pairs
        if not valid_search.any():
            search_mask = np.ones(self.ppl.shape[1], dtype=bool)
            return search_mask

        ppl_delta = (self.ppl[:-1][valid_pairs] - self.ppl[1:][valid_pairs]).reshape(-1, valid_pairs.shape[1])
        ppl_delta_thres = np.percentile(ppl_delta, self.ppl_thres_percentile, axis=0)

        ppl_search_delta =(self.ppl[:-1][valid_search] - self.ppl[1:][valid_search]).reshape(-1, valid_search.shape[1])
        ppl_search_delta = np.mean(ppl_search_delta, axis=0)

        search_mask = ppl_search_delta >= ppl_delta_thres
        return search_mask

    def write_to_file(self): 
        write_shard(self.file_path, self.pool, self.timestamps)  # Pass timestamps too

    def visualize_buffer_state(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter out infinite values for visualization
        finite_mask = np.isfinite(self.ppl[-1]) & np.isfinite(self.cr[-1])
        cr_finite = self.cr[-1, finite_mask]
        ppl_finite = self.ppl[-1, finite_mask]
        
        if len(cr_finite) == 0:
            print("No finite data to visualize")
            return None
        
        stats_text = (f"Total samples: {len(self.pool)}\n"
                     f"Finite samples: {len(cr_finite)}\n"
                     f"Mean CR: {np.mean(cr_finite):.3f}\n"
                     f"Mean PPL: {np.mean(ppl_finite):.3f}\n"
                     f"Samples below PPL threshold: {np.sum(ppl_finite < self.ppl_thres)}")
        
        plot_buffer_state(ax, cr_finite, ppl_finite, self.ppl_thres, self.ppl_thres_percentile,
                         title='Buffer State: PPL vs Control Rate', stats_text=stats_text)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        return fig
    
    def animate_buffer_evolution(self, history_log, save_path=None, interval=200):
        if not history_log:
            print("No history log provided for animation")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter finite values for limit calculation
        all_cr_finite = []
        all_ppl_finite = []
        for h in history_log:
            cr_data = h[0].cpu().numpy() if hasattr(h[0], 'cpu') else h[0]
            ppl_data = h[1].cpu().numpy() if hasattr(h[1], 'cpu') else h[1]
            finite_mask = np.isfinite(cr_data) & np.isfinite(ppl_data)
            if np.any(finite_mask):
                all_cr_finite.extend(cr_data[finite_mask])
                all_ppl_finite.extend(ppl_data[finite_mask])
        
        if not all_cr_finite:
            print("No finite data in history log")
            return None
            
        all_cr = np.array(all_cr_finite)
        all_ppl = np.array(all_ppl_finite)
        
        cr_margin = (all_cr.max() - all_cr.min()) * 0.1
        ppl_margin = (all_ppl.max() - all_ppl.min()) * 0.1
        
        xlim = (all_cr.min() - cr_margin, all_cr.max() + cr_margin)
        ylim = (all_ppl.min() - ppl_margin, all_ppl.max() + ppl_margin)
        
        def update(frame):
            plt.clf()
            ax_new = fig.add_subplot(111)
            
            cr_data, ppl_data, epoch = history_log[frame]
            cr_np = cr_data.cpu().numpy() if hasattr(cr_data, 'cpu') else cr_data
            ppl_np = ppl_data.cpu().numpy() if hasattr(ppl_data, 'cpu') else ppl_data
            
            # Filter out infinite values
            finite_mask = np.isfinite(cr_np) & np.isfinite(ppl_np)
            cr_valid = cr_np[finite_mask]
            ppl_valid = ppl_np[finite_mask]
            
            if len(cr_valid) == 0:
                ax_new.text(0.5, 0.5, f'No finite data at epoch {epoch}', 
                           transform=ax_new.transAxes, ha='center', va='center')
                return
                
            # Calculate dynamic threshold for this frame
            frame_ppl_thres = np.percentile(ppl_valid, self.ppl_thres_percentile) if len(ppl_valid) > 0 else float('inf')
            
            stats_text = (f"Epoch: {epoch}\n"
                         f"Finite samples: {len(cr_valid)}/{len(cr_np)}\n"
                         f"Mean CR: {np.mean(cr_valid):.3f}\n"
                         f"Mean PPL: {np.mean(ppl_valid):.3f}\n"
                         f"Min PPL: {np.min(ppl_valid):.3f}\n"
                         f"Samples < threshold: {np.sum(ppl_valid < frame_ppl_thres)}")
            
            plot_buffer_state(ax_new, cr_valid, ppl_valid, frame_ppl_thres, self.ppl_thres_percentile,
                            xlim=xlim, ylim=ylim, 
                            title=f'Buffer Evolution - Epoch {epoch}',
                            stats_text=stats_text, show_colorbar=True)  # Changed to True
            
            # Add mean CR vertical line
            ax_new.axvline(x=np.mean(cr_valid), color='orange', linestyle=':', alpha=0.7, 
                          label=f'Mean CR: {np.mean(cr_valid):.3f}')
            
            # Remove the text box since we now have proper colorbar
            # ax_new.text(0.02, 0.98, 'Colors: Purple=High Priority → Yellow=Low Priority', ...)
            
            if frame % 10 == 0:
                print(f"Processing frame {frame}/{len(history_log)}", end='\r')
        
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        fps = 1000 // interval
        anim = FuncAnimation(fig, update, frames=len(history_log), 
                           interval=interval, blit=False)
        
        if save_path:
            print(f"\nSaving animation to {save_path}...")
            anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
            plt.close()
            print(f"✓ Animation saved successfully!")
        else:
            plt.show()
        
        return anim
    
    def get_buffer_statistics(self):
        # Filter finite values for statistics
        finite_mask = np.isfinite(self.ppl[-1]) & np.isfinite(self.cr[-1])
        cr_finite = self.cr[-1, finite_mask]
        ppl_finite = self.ppl[-1, finite_mask]
        
        stats = {
            'total_samples': len(self.pool),
            'finite_samples': len(cr_finite),
            'infinite_samples': len(self.pool) - len(cr_finite),
            'cr_mean': np.mean(cr_finite) if len(cr_finite) > 0 else float('nan'),
            'cr_std': np.std(cr_finite) if len(cr_finite) > 0 else float('nan'),
            'cr_min': np.min(cr_finite) if len(cr_finite) > 0 else float('nan'),
            'cr_max': np.max(cr_finite) if len(cr_finite) > 0 else float('nan'),
            'ppl_mean': np.mean(ppl_finite) if len(ppl_finite) > 0 else float('nan'),
            'ppl_std': np.std(ppl_finite) if len(ppl_finite) > 0 else float('nan'),
            'ppl_min': np.min(ppl_finite) if len(ppl_finite) > 0 else float('nan'),
            'ppl_max': np.max(ppl_finite) if len(ppl_finite) > 0 else float('nan'),
            'ppl_threshold': self.ppl_thres,
            'samples_below_threshold': np.sum(ppl_finite < self.ppl_thres) if len(ppl_finite) > 0 else 0,
            'cts_mean': np.mean(self.cts[-1]),
            'cts_max': np.max(self.cts[-1]),
        }
        
        # Calculate correlation between cr and ppl (only for finite values)
        if len(cr_finite) > 1 and len(ppl_finite) > 1:
            stats['cr_ppl_correlation'] = np.corrcoef(cr_finite, ppl_finite)[0, 1]
        else:
            stats['cr_ppl_correlation'] = float('nan')
        
        return stats