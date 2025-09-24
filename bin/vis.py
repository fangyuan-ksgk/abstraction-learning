# visualize attention pattern for analysis purpose
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import get_sample_level_ppl
import numpy as np

def visualize_attn(attn_weights, batch_data, idx=0):
    """
    Visualize attention patterns with level-aware coloring and annotations.
    """
    assert idx < batch_data.batch_size, "Index out of range"
    
    sample_idx = 0
    layer_idx = 5

    # Get the sample mask for the specified index
    sample_mask = (batch_data.sample_idx == batch_data.indices[sample_idx])[:-1]  # Remove last token
    sample_positions = torch.where(sample_mask)[0]
    slice_size = len(sample_positions)

    # Get levels for this sample
    levels = batch_data.levels[:-1][sample_mask].cpu().numpy()

    # Get attention weights from first layer, first head
    attn_layer_0 = attn_weights[layer_idx]  # Shape: [batch, n_head, seq_len, seq_len]
    print(f"Attention weights shape: {attn_layer_0.shape}")
    print(f"Sample {sample_idx} has {slice_size} tokens")

    # Extract attention matrix for this sample (first head)
    attn_matrix = attn_layer_0[0, 0, :slice_size, :slice_size].detach().cpu().numpy()

    # Check if we have multiple levels
    unique_levels = np.unique(levels)
    n_levels = len(unique_levels)
    has_multiple_levels = n_levels > 1
    
    # Create figure with conditional subplots
    if has_multiple_levels:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        ax1 = axes[0]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Main attention heatmap
    im = ax1.imshow(attn_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=attn_matrix.max())
    
    if has_multiple_levels:
        # Add level boundaries and annotations for multi-level data
        level_colors = plt.cm.Set3(np.linspace(0, 1, n_levels))
        
        # Draw horizontal and vertical lines to separate levels
        for level in unique_levels:
            level_mask = levels == level
            level_indices = np.where(level_mask)[0]
            if len(level_indices) > 0:
                last_idx = level_indices[-1] + 1
                if last_idx < slice_size:
                    ax1.axhline(y=last_idx - 0.5, color='red', linewidth=1, alpha=0.5)
                    ax1.axvline(x=last_idx - 0.5, color='red', linewidth=1, alpha=0.5)
        
        # Add level annotations with colored backgrounds
        for i, level in enumerate(levels):
            color_idx = np.where(unique_levels == level)[0][0]
            color = level_colors[color_idx]
            # Left side (query)
            rect = plt.Rectangle((-2, i - 0.5), 2, 1, 
                                facecolor=color, alpha=0.3, clip_on=False)
            ax1.add_patch(rect)
            ax1.text(-1, i, f'L{level}', ha='center', va='center', fontsize=6, weight='bold')
            
            # Top side (key)  
            rect = plt.Rectangle((i - 0.5, -2), 1, 2,
                                facecolor=color, alpha=0.3, clip_on=False)
            ax1.add_patch(rect)
            ax1.text(i, -1, f'L{level}', ha='center', va='center', fontsize=6, 
                    rotation=90, weight='bold')
    else:
        # For single-level data, just add simple position labels
        print(f"Data contains only level {unique_levels[0]}")
    
    ax1.set_xlim(-0.5, slice_size - 0.5)
    ax1.set_ylim(slice_size - 0.5, -0.5)
    ax1.set_xlabel('Key Position', fontsize=12)
    ax1.set_ylabel('Query Position', fontsize=12)
    
    if has_multiple_levels:
        ax1.set_title(f'Attention Pattern (Layer {layer_idx}, Head 0) - Sample {sample_idx}', fontsize=14)
    else:
        ax1.set_title(f'Attention Pattern (Layer {layer_idx}, Head 0) - Sample {sample_idx} - Level {unique_levels[0]} only', fontsize=14)
    
    # Add colorbar
    plt.colorbar(im, ax=ax1, label='Attention Weight')
    
    # Level-wise attention aggregation (only if multiple levels)
    if has_multiple_levels:
        ax2 = axes[1]
        level_attn_matrix = np.zeros((n_levels, n_levels))
        
        for i, level_q in enumerate(unique_levels):
            for j, level_k in enumerate(unique_levels):
                mask_q = levels == level_q
                mask_k = levels == level_k
                if np.any(mask_q) and np.any(mask_k):
                    level_attn_matrix[i, j] = attn_matrix[np.ix_(mask_q, mask_k)].mean()
        
        im2 = ax2.imshow(level_attn_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0)
        
        # Add value annotations
        for i in range(n_levels):
            for j in range(n_levels):
                # Choose text color based on background intensity
                text_color = "white" if level_attn_matrix[i, j] > level_attn_matrix.max() * 0.5 else "black"
                text = ax2.text(j, i, f'{level_attn_matrix[i, j]:.3f}',
                                ha="center", va="center", color=text_color, fontsize=30)
        
        ax2.set_xticks(range(n_levels))
        ax2.set_yticks(range(n_levels))
        ax2.set_xticklabels([f'Level {l}' for l in unique_levels])
        ax2.set_yticklabels([f'Level {l}' for l in unique_levels])
        ax2.set_xlabel('Key Level', fontsize=12)
        ax2.set_ylabel('Query Level', fontsize=12)
        ax2.set_title('Average Attention Between Levels', fontsize=14)
        plt.colorbar(im2, ax=ax2, label='Average Attention')
    
    plt.tight_layout()
    plt.show()


# visualize the dynamic of 'searching' for each sample
# --------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def visualize_backtrack(critical_timesteps, sample_idx, sample_timestamps, sample_ppt, ppl_thres):
    plt.figure(figsize=(12, 6))
    plt.plot(sample_timestamps.cpu().numpy(), sample_ppt.cpu().numpy(), 'b-', linewidth=2, label='Perplexity')
    plt.axhline(y=ppl_thres, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({ppl_thres:.2f})')
    
    # Highlight critical timestamps
    critical_mask = sample_ppt > ppl_thres
    if critical_mask.any():
        critical_ts = sample_timestamps[critical_mask]
        critical_ppt = sample_ppt[critical_mask]
        plt.scatter(critical_ts.cpu().numpy(), critical_ppt.cpu().numpy(), 
                   color='red', s=100, zorder=5, label='Critical Points')
    
    # Mark backtrack point
    if sample_idx < len(critical_timesteps):
        backtrack_ts = critical_timesteps[sample_idx]
        if backtrack_ts > 0:
            plt.axvline(x=backtrack_ts, color='green', linestyle=':', 
                       linewidth=2, label=f'Backtrack (t={backtrack_ts})')
    
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(f'Backtrack Visualization - Sample {sample_idx}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

import io
from PIL import Image

def visualize_multi_sample_backtrack(batch_data, ppt, cts, buffer, num_samples=4, figsize=(16, 12)):
    """
    Create a grid visualization of backtrack analysis for multiple samples.
    Returns PIL Image without displaying.
    """
    from utils import get_sample_level_ppl
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    rows = grid_size
    cols = grid_size if num_samples > (grid_size - 1) * grid_size else grid_size - 1
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'Backtrack Visualization for {num_samples} Samples', fontsize=16)
    
    if num_samples == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()
    
    actual_samples = min(num_samples, batch_data.indices.max().item() + 1)
    
    for i in range(actual_samples):
        sample_idx = i
        per_sample_ppt, per_sample_timestamps, per_sample_max_abs_ts = get_sample_level_ppl(batch_data, ppt, level=0)
        max_abs_ts = per_sample_max_abs_ts[sample_idx]
        sample_timestamps = per_sample_timestamps[sample_idx][:max_abs_ts + batch_data.K]
        sample_ppt = per_sample_ppt[sample_idx][:max_abs_ts + batch_data.K]
        ax = axes_flat[i]
        
        ax.plot(sample_timestamps.cpu().numpy(), sample_ppt.cpu().numpy(), 'b-', linewidth=1.5, label='Perplexity')
        ax.axhline(y=buffer.ppl_thres, color='r', linestyle='--', linewidth=1, label=f'Threshold ({buffer.ppl_thres:.2f})')
        
        if sample_idx < len(cts):
            backtrack_ts = cts[sample_idx] + 1
            if backtrack_ts > 0:
                ax.axvline(x=backtrack_ts, color='green', linestyle=':', linewidth=1.5, label=f'Backtrack (t={backtrack_ts})')
                backtrack_idx = (sample_timestamps == backtrack_ts).nonzero(as_tuple=True)[0]
                if len(backtrack_idx) > 0:
                    backtrack_y = sample_ppt[backtrack_idx[0]]
                    ax.scatter(backtrack_ts, backtrack_y.cpu().numpy(), marker='^', color='green', s=80, zorder=5, label='Backtrack Points')

        critical_mask = (sample_ppt > buffer.ppl_thres) & (sample_timestamps != backtrack_ts)
        if critical_mask.any():
            critical_ts = sample_timestamps[critical_mask]
            critical_ppt = sample_ppt[critical_mask]
            ax.scatter(critical_ts.cpu().numpy(), critical_ppt.cpu().numpy(), color='red', s=50, zorder=5, label='Critical Points')
            
        ax.set_xlabel('Timestamp', fontsize=10)
        ax.set_ylabel('Perplexity', fontsize=10)
        ax.set_title(f'Sample {sample_idx}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim([2.0, 5.5])
        ax.set_xlim([0, 100])
    
    for i in range(actual_samples, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Use tight_layout with figure object to avoid display
    fig.tight_layout()
    
    # Save to buffer and convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    pil_image = Image.open(buf)
    
    # Important: Close the figure to free memory and prevent display
    plt.close(fig)
    
    return pil_image