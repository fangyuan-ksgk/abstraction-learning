import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, List
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import re

# Visualizer function for n-body system
# --------------------------------------------------------------------------------------------------------------------------
def vis_nbody(
    positions: np.ndarray,
    masses: Optional[np.ndarray] = None,
    forces: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "N-Body System",
    figsize: Tuple[int, int] = (12, 10),
    view_angle: Tuple[float, float] = (25, 45),
    show_forces: bool = True,
    force_scale: float = 0.3,
    mass_scale: float = 1.0,
    legend_scale: float = 1.0,
    body_colors: Optional[List[str]] = None,
    force_color: str = '#00FFFF',
    dark_theme: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Static visualizer of n-body system"""

    n_bodies = len(positions)
    if positions.shape[1] == 2:
        positions = np.column_stack([positions, np.zeros(n_bodies)])
    
    masses = masses if masses is not None else np.ones(n_bodies)
    mass_sizes = 200 + 1000 * (masses / masses.max())**0.5
    mass_sizes = mass_sizes * mass_scale
    
    if body_colors is None:
        cmap, norm = plt.cm.plasma, plt.Normalize(vmin=masses.min(), vmax=masses.max())
        body_colors = [cmap(norm(m)) for m in masses]
    
    labels = labels if labels is not None else [f"Body {i+1}" for i in range(n_bodies)]
    
    plt.style.use('dark_background' if dark_theme else 'default')
    bg_color = 'black' if dark_theme else 'white'
    text_color = 'white' if dark_theme else 'black'
    
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_bodies):
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2],
                  c=[body_colors[i]], s=mass_sizes[i], 
                  edgecolors='white' if dark_theme else 'black',
                  linewidth=0.5, alpha=0.9, zorder=10)
    
    if show_forces and forces is not None:
        for i in range(n_bodies):
            assert forces.ndim == 2, "forces must be 2D array, (n_bodies, 3)"
            ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                        forces[i, 0] * force_scale, forces[i, 1] * force_scale,
                        forces[i, 2] * force_scale, color=force_color, linewidth=2.5,
                        arrow_length_ratio=0.2, alpha=0.9)
    
    pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
    center = (pos_min + pos_max) / 2
    max_range = np.max(pos_max - pos_min) * 0.6
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.set_title(title, color=text_color, fontsize=16, pad=15)
    
    legend_elements = [Line2D([], [], marker='o', color='w' if dark_theme else 'black',
                              markerfacecolor=body_colors[i], markersize=10,
                              linestyle='', label=labels[i]) for i in range(min(n_bodies, 5))]
    if show_forces and forces is not None:
        legend_elements.append(Line2D([], [], color=force_color, linewidth=2, label='Forces'))
    
    ax.legend(handles=legend_elements, loc='upper center', framealpha=0.9, fontsize=10*legend_scale, ncol=len(legend_elements))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
    
    return fig



def create_nbody_gif(times, positions, forces=None, masses=None, labels=None,
                     body_colors=None, save_path='nbody.gif', fps=30, trail_steps=30,
                     force_scale=0.01, mass_scale=1.0, legend_scale=1.0,
                     dark_theme=False, title='N-Body System', **kwargs):
    """Create N-body GIF with vis_nbody style, fixed scale, and trajectories."""
    n_frames, n_bodies = len(times), positions.shape[1]
    masses = masses if masses is not None else np.ones(n_bodies)
    labels = labels if labels is not None else [f'Body {i+1}' for i in range(n_bodies)]
    
    all_pos = positions.reshape(-1, 3)
    center = all_pos.mean(axis=0)
    max_range = np.ptp(all_pos, axis=0).max() * 0.6
    xlim, ylim, zlim = [(center[i] - max_range, center[i] + max_range) for i in range(3)]
    
    fig = plt.figure(figsize=(12, 10))
    
    def update(frame):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        
        if trail_steps > 0 and frame > 0:
            trail_start = max(0, frame - trail_steps)
            for i in range(n_bodies):
                trail = positions[trail_start:frame+1, i]
                for j in range(len(trail)-1):
                    alpha = 0.2 + 0.4 * (j / max(len(trail)-1, 1))
                    ax.plot(trail[j:j+2, 0], trail[j:j+2, 1], trail[j:j+2, 2],
                           color=body_colors[i] if body_colors else f'C{i}',
                           alpha=alpha, linewidth=1.5, zorder=1)
        
        mass_sizes = 200 + 1000 * (masses / masses.max())**0.5 * mass_scale
        body_colors_frame = body_colors if body_colors else [plt.cm.plasma(plt.Normalize(masses.min(), masses.max())(m)) for m in masses]
        
        for i in range(n_bodies):
            ax.scatter(*positions[frame, i], c=[body_colors_frame[i]], s=mass_sizes[i],
                      edgecolors='black', linewidth=0.5, alpha=0.9, zorder=10)
        
        if forces is not None:
            for i in range(n_bodies):
                if np.linalg.norm(forces[frame, i]) > 1e-10:
                    ax.quiver(*positions[frame, i], *(forces[frame, i] * force_scale),
                             color='#00FFFF', linewidth=2.5, arrow_length_ratio=0.2, alpha=0.9)
        
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.view_init(elev=25, azim=45)
        ax.grid(False)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f'{title} | t = {times[frame]:.2f}', color='black', fontsize=16, pad=15)
        
        legend_elements = [Line2D([], [], marker='o', color='black', markerfacecolor=body_colors_frame[i],
                                 markersize=10, linestyle='', label=labels[i]) for i in range(min(n_bodies, 5))]
        if forces is not None:
            legend_elements.append(Line2D([], [], color='#00FFFF', linewidth=2, label='Forces'))
        ax.legend(handles=legend_elements, loc='upper center', framealpha=0.9, 
                 fontsize=10*legend_scale, ncol=len(legend_elements))
        
        if frame % 10 == 0:
            print(f"Processing frame {frame}/{n_frames}", end='\r')
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    print(f"\nSaving GIF to {save_path}...")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close()
    print(f"✓ GIF saved successfully!")
    return save_path


def visualize_nbody_comparison(
    positions: np.ndarray,
    forces_true: np.ndarray,
    forces_pred: np.ndarray,
    masses: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    view_angle: Tuple[float, float] = (20, 30),
    force_scale: float = 0.3,
    body_colors: Optional[List[str]] = None,
    dark_theme: bool = True,
    save_path: Optional[str] = None,
    show_axes_planes: bool = False,  # Added to match visualize_nbody_system
    force_colors_true: Optional[List[str]] = None,  # Individual force colors
    force_colors_pred: Optional[List[str]] = None,  # Individual force colors
    remove_body_edges: bool = True  # Added to match visualize_nbody_system
) -> plt.Figure:
    """
    Visualize comparison between true and predicted forces in an N-body system.
    Now aligned with visualize_nbody_system parameters and style.
    """
    n_bodies = len(positions)
    if positions.shape[1] == 2:
        positions = np.column_stack([positions, np.zeros(n_bodies)])
        forces_true = np.column_stack([forces_true, np.zeros(n_bodies)])
        forces_pred = np.column_stack([forces_pred, np.zeros(n_bodies)])
    
    masses = masses if masses is not None else np.ones(n_bodies)
    mass_sizes = 200 + 1000 * (masses / masses.max())**0.5
    
    if body_colors is None:
        cmap, norm = plt.cm.plasma, plt.Normalize(vmin=masses.min(), vmax=masses.max())
        body_colors = [cmap(norm(m)) for m in masses]
    
    labels = labels if labels is not None else [f"Body {i+1}" for i in range(n_bodies)]
    
    # Default force colors if not provided
    if force_colors_true is None:
        force_colors_true = ['#00FFFF'] * n_bodies  # Cyan for true
    if force_colors_pred is None:
        force_colors_pred = ['#FF1493'] * n_bodies  # Pink for predicted
    
    plt.style.use('dark_background' if dark_theme else 'default')
    bg_color = 'black' if dark_theme else 'white'
    text_color = 'white' if dark_theme else 'black'
    
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    for ax, force_data, force_colors, subplot_title in [
        (ax1, forces_true, force_colors_true, 'True Forces'),
        (ax2, forces_pred, force_colors_pred, 'Predicted Forces')
    ]:
        # Plot bodies (matching visualize_nbody_system style)
        for i in range(n_bodies):
            # Determine edge color based on remove_body_edges parameter
            edge_color = 'none' if remove_body_edges else ('white' if dark_theme else 'black')
            
            ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2],
                      c=[body_colors[i]], s=mass_sizes[i], 
                      edgecolors=edge_color,
                      linewidth=0 if remove_body_edges else 0.5, 
                      alpha=0.9, zorder=10)
        
        # Plot forces (matching visualize_nbody_system style)
        for i in range(n_bodies):
            if np.linalg.norm(force_data[i]) > 1e-10:
                ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                         force_data[i, 0] * force_scale, 
                         force_data[i, 1] * force_scale,
                         force_data[i, 2] * force_scale, 
                         color=force_colors[i], 
                         linewidth=2.5,
                         arrow_length_ratio=0.2, alpha=0.9)
        
        # Set limits (matching visualize_nbody_system logic)
        pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
        center = (pos_min + pos_max) / 2
        max_range = np.max(pos_max - pos_min) * 0.6
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Handle axes visibility (matching visualize_nbody_system)
        if not show_axes_planes:
            # Remove all visual elements of the axes
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.line.set_linewidth(0)
            ax.yaxis.line.set_linewidth(0)
            ax.zaxis.line.set_linewidth(0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
        else:
            # Show axes with labels
            ax.grid(True, alpha=0.2)
            ax.set_xlabel('X (AU)', color=text_color, fontsize=9)
            ax.set_ylabel('Y (AU)', color=text_color, fontsize=9)
            ax.set_zlabel('Z (AU)', color=text_color, fontsize=9)
        
        # Title
        ax.set_title(subplot_title, color=text_color, fontsize=14, pad=15)
        
        # Legend (only on first subplot to avoid duplication)
        if ax == ax1:
            legend_elements = [Line2D([], [], marker='o', color='w' if dark_theme else 'black',
                                     markerfacecolor=body_colors[i], markersize=10,
                                     linestyle='', label=labels[i]) for i in range(min(n_bodies, 5))]
            legend_elements.append(Line2D([], [], color=force_colors_true[0], 
                                         linewidth=2, label='True Forces'))
            ax.legend(handles=legend_elements, loc='upper left', framealpha=0.8, fontsize=10)
        else:
            # Add force legend for predicted
            legend_elements = [Line2D([], [], color=force_colors_pred[0], 
                                     linewidth=2, label='Predicted Forces')]
            ax.legend(handles=legend_elements, loc='upper left', framealpha=0.8, fontsize=10)
    
    # Add error metrics at the bottom
    force_errors = np.linalg.norm(forces_pred - forces_true, axis=1)
    mean_error = np.mean(force_errors)
    max_error = np.max(force_errors)
    rel_error = np.mean(force_errors) / (np.mean(np.linalg.norm(forces_true, axis=1)) + 1e-10) * 100
    
    error_text = f'Mean Error: {mean_error:.3e}  |  Max Error: {max_error:.3e}  |  Relative Error: {rel_error:.1f}%'
    fig.text(0.5, 0.02, error_text, ha='center', fontsize=11,
             color=text_color,
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor=bg_color, 
                      edgecolor=text_color,
                      alpha=0.8))
    
    # Main title
    fig.suptitle('N-Body System: Force Comparison', 
                fontsize=16, fontweight='bold',
                color=text_color, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
    
    return fig

# --------------------------------------------------------------------------------------------------------------------------





import rebound 


def create_random_cartesian(n=3, seed=42, pos_range=2.0, vel_range=0.3, mass_range=(0.5, 2.0)):
    """Random n-body using Cartesian coordinates (position/velocity)."""
    np.random.seed(seed)
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    for i in range(n):
        sim.add(m=np.random.uniform(*mass_range),
                x=np.random.uniform(-pos_range, pos_range),
                y=np.random.uniform(-pos_range, pos_range),
                z=np.random.uniform(-pos_range/4, pos_range/4),
                vx=np.random.uniform(-vel_range, vel_range),
                vy=np.random.uniform(-vel_range, vel_range),
                vz=np.random.uniform(-vel_range/3, vel_range/3))
    
    sim.move_to_com()
    return sim


def create_random_orbital(n=3, seed=42, a_range=(0.5, 5.0), e_range=(0, 0.3), mass_range=(0.1, 1.0)):
    """Random n-body using orbital elements (hierarchical system)."""
    np.random.seed(seed)
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # Central massive body
    sim.add(m=np.random.uniform(1.0, 3.0))
    
    # Add orbiting bodies
    for i in range(n-1):
        sim.add(m=np.random.uniform(*mass_range),
                a=np.random.uniform(*a_range),
                e=np.random.uniform(*e_range),
                inc=np.random.uniform(0, 0.1),  # Mostly planar
                Omega=np.random.uniform(0, 2*np.pi),
                omega=np.random.uniform(0, 2*np.pi),
                f=np.random.uniform(0, 2*np.pi))
    
    sim.move_to_com()
    return sim


# N-body data pre-processing
# --------------------------------------------------------------------------------------------------------------------------

def prep_pretrain_minimal(raw_data: Tuple, n_context: int = 5, include_masses: bool = True, round_to: int = 2, answer_token: str = '') -> List[str]:
    """Formatting function, optionally include answer token"""

    times, positions, _ = raw_data
    n_bodies = positions.shape[1]
    masses = np.ones(n_bodies)  # Default equal masses
    
    sequences = []
    for i in range(len(times) - n_context):
        lines = [f"M|{'|'.join([f'{m:.2f}' for m in masses])}"] if include_masses else []
        lines += [f"{times[j]:.2f}|{'|'.join([f'{p[0]:.{round_to}f},{p[1]:.{round_to}f},{p[2]:.{round_to}f}' for p in positions[j]])}" 
                for j in range(i, i + n_context)]
        lines += [f"{times[i+n_context]:.2f}|{answer_token}{'|'.join([f'{p[0]:.{round_to}f},{p[1]:.{round_to}f},{p[2]:.{round_to}f}' for p in positions[i+n_context]])}"]
        sequences.append("\n".join(lines))
        
    return sequences

def retrieve_nbody_data(sequences: List[str]): 
    times, positions = [], [] 
    for seq in sequences: 
        lines = seq.split("\n")
        masses = [float(m) for m in lines[0].split("|")[1:]]

        t = [float(t.split("|")[0]) for t in lines[1:]]
        pos = [tuple(map(float, p.split("|")[1:])) for p in lines[1:]]

        times.append(t)
        positions.append(pos)

    return times, positions, masses


def simulate_and_extract_data(sim, t_max=100, n_points=1001, stride=1):
    times = np.linspace(0, t_max, n_points // stride)
    n, G, positions, forces = len(sim.particles), sim.G, [], []
    
    for t in times:
        sim.integrate(t)
        pos = np.array([[p.x, p.y, p.z] for p in sim.particles])
        positions.append(pos)
        force = np.zeros((n, 3))
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 0:
                        force[i] += G * sim.particles[i].m * sim.particles[j].m * r_vec / r_mag**3
        forces.append(force)
    
    return times, np.array(positions), np.array(forces)


def create_dataset_with_params(patterns=None, n_bodies=3, n_context=5, T=100, include_masses=True, stride=1,
                               round_to=2, answer_token=''):
    
    patterns = patterns or ['cartesian', 'orbital']
    all_sequences, metadata = [], []
    
    for pattern in patterns:
        print(f"Generating {pattern} with n={n_bodies}, n_context={n_context}...")
        n_instances = 10
        
        for instance in range(n_instances):
            sim = (create_random_cartesian(n=n_bodies, seed=np.random.randint(10000)) 
                   if pattern == 'cartesian' else 
                   create_random_orbital(n=n_bodies, seed=np.random.randint(10000)))
            
            raw_data = simulate_and_extract_data(sim, t_max=T, n_points= T*10 + stride, stride=stride)

            sequences = prep_pretrain_minimal(raw_data, n_context, include_masses, round_to, answer_token)
            
            all_sequences.extend(sequences)
            metadata.extend([{'pattern': pattern, 'n_bodies': n_bodies, 'instance': instance,
                            'n_context': n_context, 'has_masses': include_masses}] * len(sequences))
    
    return {'sequences': all_sequences, 'metadata': metadata,
            'config': {'n_bodies': n_bodies, 'n_context': n_context, 
                      'include_masses': include_masses, 'patterns': patterns,
                      'answer_token': answer_token}}

# from constant import MASK_TOK
# MASK_TOK = '<mask>'
ANSWER_TOK = '<a>'

class TinyTokenizer: 
    def __init__(self):
        chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                 '.', '|', 'M', '\n', '-', ',', ANSWER_TOK]
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.inverse_vocab = {i: char for char, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.answer_token = ANSWER_TOK
        self.answer_token_id = self.vocab[self.answer_token]

        pat = sorted(self.vocab.keys(), key=len, reverse=True)
        self.pattern = re.compile('|'.join(re.escape(p) for p in pat))

    def encode(self, seq: str):
        tokens = self.pattern.findall(seq)
        return [self.vocab[t] for t in tokens]
    
    def decode(self, tokens: list):
        return ''.join([self.inverse_vocab[t] for t in tokens if t in self.inverse_vocab])
    
    def __call__(self, seq: str):
        return self.encode(seq)





# Example Code
# --------------------------------------------------------------------------------------------------------------------------

def simulate(): 
    sim = create_random_orbital(n=6, seed=1)
    # sim = create_random_cartesian(n=6, seed=42)

    raw_data = simulate_and_extract_data(sim, t_max=30, n_points=301)

    # Visualize dynamics (Now make em dance!)
    # Six body
    ts, positions, forces = raw_data
    masses = np.array([p.m for p in sim.particles])
    create_nbody_gif(ts, positions, None, masses,
                    labels=['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta'],
                    body_colors=['#FFFF00', '#00FF00', '#0000FF', '#FF0000', '#0000FF', '#00FF00'],
                    figsize=(10, 8),
                    legend_scale=1.6,
                    trail_steps=30,  # Show last 30 timesteps as trail
                    force_scale=0.001,
                    save_path='6body.gif')



# Datset Class 
# --------------------------------------------------------------------------------------------------------------------------
from dataset.base import BaseDataset, compute_hier_seq_len
from dataclasses import dataclass, field
from typing import List
import numpy as np
import re
from matplotlib import pyplot as plt
from typing import List, Optional, Tuple


@dataclass 
class NBodyDataset(BaseDataset): 
    n_bodies: int = 2
    patterns: List[str] = field(default_factory=lambda: ['cartesian'])
    n_context: int = 3
    stride: int = 1
    T: int = 10
    include_masses: bool = True
    K: int = 2
    L: int = 3
    # filepath: str = 'dataset/nbody/sequences.bin'
    tokenizer: TinyTokenizer = TinyTokenizer()
    answer_token: str = TinyTokenizer().answer_token
    answer_token_id: int = TinyTokenizer().answer_token_id

    def build(self): 
        # Generate raw data
        dataset = create_dataset_with_params(
            n_bodies=self.n_bodies,
            patterns=self.patterns,
            n_context=self.n_context,
            stride=self.stride,
            T=self.T,
            include_masses=self.include_masses,
            answer_token=self.answer_token,
        )
        
        # Tokenize sequences
        self.vocab_size = self.tokenizer.vocab_size
        self.sequences = [self.tokenizer(s) for s in dataset['sequences']]
        self.lengths = [compute_hier_seq_len(seq, self.L, self.K) for seq in self.sequences]

        if self.num_data is not None:
            self.sequences = self.sequences[:self.num_data]
            self.lengths = self.lengths[:self.num_data]
        
        # Save to disk
        self._save()
        return self