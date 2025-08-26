# visualize attention pattern for analysis purpose
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import torch

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