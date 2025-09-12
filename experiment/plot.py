import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def parse_log_file(filepath="log.txt"):
    """
    Parses the log file to extract SSL and GRPO losses.
    """
    sorl_ssl_losses = []
    ssl_only_losses = []
    sorl_grpo_losses = [] # New list for GRPO losses
    
    current_mode = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "SoRL (GAT)" in line:
                    current_mode = "SORL"
                elif "SSl (Transformer)" in line:
                    current_mode = "SSL_ONLY"
                
                if current_mode == "SORL":
                    # Extract both SSL and GRPO loss for SoRL
                    ssl_match = re.search(r"ssl_loss:\s*(-?\d+\.\d+)", line)
                    grpo_match = re.search(r"grpo_loss:\s*(-?\d+\.\d+)", line)
                    if ssl_match:
                        sorl_ssl_losses.append(float(ssl_match.group(1)))
                    if grpo_match:
                        sorl_grpo_losses.append(float(grpo_match.group(1)))

                elif current_mode == "SSL_ONLY":
                    ssl_match = re.search(r"ssl_loss:\s*(-?\d+\.\d+)", line)
                    if ssl_match:
                        ssl_only_losses.append(float(ssl_match.group(1)))

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None, None
        
    return sorl_ssl_losses, ssl_only_losses, sorl_grpo_losses


def find_crossover_point(sorl_losses, ssl_only_losses, window_size=10):
    """Find cross-over point in learning efficiency"""
    raw_sorl_gradients = - np.diff(sorl_losses)
    raw_ssl_gradients = - np.diff(ssl_only_losses)

    window_size = 20 

    for i in range(len(raw_sorl_gradients)):
        if i < window_size // 2 or i + window_size // 2 >= len(raw_sorl_gradients):
            continue 
        sorl_gradient = raw_sorl_gradients[i - window_size // 2: i + window_size // 2].mean()
        ssl_gradient = raw_ssl_gradients[i - window_size // 2: i + window_size // 2].mean()
        if sorl_gradient > ssl_gradient:
            print(f"Crossover point found at iteration {i}")
            break 

    ratio_pre_crossover = raw_sorl_gradients[:i].sum() / raw_ssl_gradients[:i].sum()
    ratio_post_crossover = raw_sorl_gradients[i:].sum() / raw_ssl_gradients[i:].sum()

    return i, ratio_pre_crossover, ratio_post_crossover

def plot_learning_efficiency(sorl_losses, ssl_only_losses, sorl_grpo_losses, crossover_point, ratio_pre_crossover, ratio_post_crossover):
    """
    Plots the learning efficiency comparison with cumulative policy improvement.
    """
    if not sorl_losses or not ssl_only_losses:
        print("Loss data is empty. Cannot generate plot.")
        return

    num_updates = len(sorl_losses)
    x_axis = np.arange(1, num_updates + 1)

    # Create figure with primary axis
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot SSL losses on primary (left) y-axis
    ax1.plot(x_axis, ssl_only_losses, label="SSL | Transformer", color='royalblue', linewidth=2)
    ax1.plot(x_axis, sorl_losses, label=f"SoRL (SSL + GRPO) | Generative Abstraction Transformer (GAT)", color='darkorange', linewidth=2)
    ax1.set_xlabel("Number of Iterations", fontsize=12)
    ax1.set_ylabel("SSL Loss (Lower is Better)", fontsize=12, color='black')
    ax1.set_xlim(0, num_updates)
    ax1.set_ylim(0)
    
    # Create secondary (right) y-axis for Cumulative Policy Improvement
    if sorl_grpo_losses:
        ax2 = ax1.twinx()
        # Calculate cumulative policy improvement (negative GRPO loss = improvement)
        cumulative_improvement = np.cumsum(-np.array(sorl_grpo_losses))
        ax2.plot(x_axis, cumulative_improvement, label="SoRL (Cumulative Policy Improvement)", 
                color='green', linestyle=':', linewidth=2.5)
        ax2.set_ylabel("Cumulative Policy Improvement (Higher is Better)", fontsize=12, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(bottom=0, top=cumulative_improvement.max() * 1.3)  # Start from 0 for cumulative improvement

    if crossover_point:
        ax1.axvline(x=crossover_point, color='crimson', linestyle='--', linewidth=1.5, 
                    label=f'Efficiency Crossover (Iteration {crossover_point})')

        # Add phase labels with ratios
        phase1_center = crossover_point / 2
        phase2_center = crossover_point + (num_updates - crossover_point) / 2

        ax1.text(phase1_center, 3.0, f'Phase 1: Search\n(SSL is more efficient)\nSoRL / SSL Efficiency Ratio: {ratio_pre_crossover:.2f}', 
                 horizontalalignment='center', color='royalblue', fontsize=11)
        ax1.text(phase2_center, 1.5, f'Phase 2: Acceleration\n(SoRL accelerates)\nSoRL / SSL Efficiency Ratio: {ratio_post_crossover:.2f}', 
                 horizontalalignment='center', color='darkorange', fontsize=11)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if sorl_grpo_losses:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    else:
        ax1.legend(lines1, labels1, fontsize=11)
    
    plt.title("SoRL Dynamics: Acceleration of Learning", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sorl_losses, ssl_only_losses, sorl_grpo_losses = parse_log_file("log.txt")

    if sorl_losses and ssl_only_losses:
        # Define the window size for the running average
        smoothing_window = 10
        
        # 1. Find the precise crossover point using the smoothed gradients
        crossover, ratio_pre_crossover, ratio_post_crossover = find_crossover_point(sorl_losses, ssl_only_losses, window_size=smoothing_window)

        # 2. Generate the visualization with cumulative policy improvement
        plot_learning_efficiency(sorl_losses, ssl_only_losses, sorl_grpo_losses, crossover, ratio_pre_crossover, ratio_post_crossover)