import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, List
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

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





# N-body data generation
# ------------------------------------------------------------------------------------------------
import rebound 

def create_random_n_body(n=3, seed=42, com_center=True): 
    """
    Hyper-parameter tunable: range of position/velocity/mass
    """
    np.random.seed(seed)
    sim = rebound.Simulation() 
    sim.units = ('yr', 'AU', 'Msun')
    for i in range(n): 
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(-0.5, 0.5)  # Mostly planar
        
        vx = np.random.uniform(-0.3, 0.3)
        vy = np.random.uniform(-0.3, 0.3)
        vz = np.random.uniform(-0.1, 0.1)
        
        mass = np.random.uniform(0.8, 10.0)
        
        sim.add(m=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    if com_center: 
        sim.move_to_com() # center of mass as origin (relative position/velocity w.r.t. CoM presented)
    
    return sim


def create_circular_binary(m1=1.0, m2=1.0, separation=1.0, com_center=True):
    """Create two bodies in circular orbit around their center of mass."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # Calculate orbital velocity for circular orbit
    total_mass = m1 + m2
    r1 = separation * m2 / total_mass  # Distance of m1 from COM
    r2 = separation * m1 / total_mass  # Distance of m2 from COM
    
    # Orbital velocity: v = sqrt(G*M/r) for reduced mass system
    v1 = np.sqrt(sim.G * m2**2 / (total_mass * separation))
    v2 = np.sqrt(sim.G * m1**2 / (total_mass * separation))
    
    sim.add(m=m1, x=-r1, y=0, z=0, vx=0, vy=v1, vz=0)
    sim.add(m=m2, x=r2, y=0, z=0, vx=0, vy=-v2, vz=0)
    
    if com_center:
        sim.move_to_com()
    
    return sim


def create_hierarchical_triple(m_central=1.0, m_inner=0.1, m_outer=0.05, 
                               a_inner=0.5, a_outer=3.0, com_center=True):
    """Create stable hierarchical triple: central star with close and distant planets."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    sim.add(m=m_central)  # Central star
    sim.add(m=m_inner, a=a_inner, e=0.01)  # Inner planet (circular)
    sim.add(m=m_outer, a=a_outer, e=0.02)  # Outer planet (circular)
    
    if com_center:
        sim.move_to_com()
    
    return sim


def create_figure8_3body(high_precision=True):
    """Create figure-8 with optional high precision."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    if high_precision:
        # Use IAS15 integrator (15th order, adaptive timestep)
        sim.integrator = "ias15"
        sim.ri_ias15.epsilon = 1e-12  # Very high precision
    else:
        # Default WHFast (lower precision but faster)
        sim.integrator = "whfast"
        sim.dt = 0.001
    
    # Ultra-precise initial conditions (more decimal places)
    sim.add(m=1.0, 
            x=-0.97000436, 
            y=0.24308753, 
            z=0,
            vx=-0.466203685/2, 
            vy=-0.43236573/2, 
            vz=0)
    sim.add(m=1.0, 
            x=0, 
            y=0, 
            z=0,
            vx=0.466203685, 
            vy=0.43236573, 
            vz=0)
    sim.add(m=1.0, 
            x=0.97000436, 
            y=-0.24308753, 
            z=0,
            vx=-0.466203685/2, 
            vy=-0.43236573/2, 
            vz=0)
    
    sim.move_to_com()
    return sim


def create_figure8_3body_precise():
    """Ultra-precise figure-8 initial conditions from Simó (2002)."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.integrator = "ias15"
    sim.ri_ias15.epsilon = 1e-14
    
    # These are the most precise known values
    x1 = -0.9700436
    y1 = 0.24308753
    vx = 0.4662036850
    vy = 0.4323657300
    
    sim.add(m=1.0, x=x1, y=y1, z=0, vx=vx/2, vy=vy/2, vz=0)
    sim.add(m=1.0, x=-x1, y=-y1, z=0, vx=vx/2, vy=vy/2, vz=0)
    sim.add(m=1.0, x=0, y=0, z=0, vx=-vx, vy=-vy, vz=0)
    
    sim.move_to_com()
    return sim


def create_lagrange_triangle(m1=1.0, m2=1.0, m3=1.0, radius=1.0, com_center=True):
    """Create three equal masses in stable Lagrange configuration."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # First, add two bodies in circular orbit
    sim.add(m=m1, x=-radius/2, y=0, z=0)
    sim.add(m=m2, x=radius/2, y=0, z=0)
    
    # Calculate their mutual orbital velocity
    v_orbit = np.sqrt(sim.G * (m1 + m2) / radius)
    sim.particles[0].vy = v_orbit * m2/(m1+m2)
    sim.particles[1].vy = -v_orbit * m1/(m1+m2)
    
    # Add third body at the L4 Lagrange point (60° ahead)
    # L4 is at equal distance from both bodies
    x3 = radius * np.cos(np.pi/3) - radius/2
    y3 = radius * np.sin(np.pi/3)
    
    # For equal masses, the third body needs the same angular velocity
    # Calculate velocity for circular motion around COM
    r3_from_com = radius / np.sqrt(3)  # Distance from COM for equal masses
    v3_mag = np.sqrt(sim.G * (m1 + m2 + m3) / (radius * np.sqrt(3)))
    
    # Velocity perpendicular to radius from COM
    angle3 = np.arctan2(y3, x3 + radius/(2*(1 + m2/m1)))
    vx3 = -v3_mag * np.sin(angle3)
    vy3 = v3_mag * np.cos(angle3)
    
    sim.add(m=m3, x=x3, y=y3, z=0, vx=vx3, vy=vy3, vz=0)
    
    if com_center:
        sim.move_to_com()
    
    return sim


def create_pythagorean_3body():
    """Create Pythagorean 3-body problem: masses in 3:4:5 ratio, zero initial velocity."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # Classic Pythagorean problem setup
    sim.add(m=3.0, x=1.0, y=3.0, z=0, vx=0, vy=0, vz=0)
    sim.add(m=4.0, x=-2.0, y=-1.0, z=0, vx=0, vy=0, vz=0)
    sim.add(m=5.0, x=1.0, y=-1.0, z=0, vx=0, vy=0, vz=0)
    
    sim.move_to_com()
    return sim


def create_sitnikov_problem(m_binary=1.0, separation=1.0, m_test=0.001, z0=1.0):
    """Create Sitnikov problem: test mass oscillating perpendicular to binary."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # Binary in x-y plane
    v_binary = np.sqrt(sim.G * m_binary / (2 * separation))
    sim.add(m=m_binary, x=-separation/2, y=0, z=0, vx=0, vy=v_binary, vz=0)
    sim.add(m=m_binary, x=separation/2, y=0, z=0, vx=0, vy=-v_binary, vz=0)
    
    # Test particle on z-axis
    sim.add(m=m_test, x=0, y=0, z=z0, vx=0, vy=0, vz=0)
    
    sim.move_to_com()
    return sim


def create_sun_earth_moon():
    """Create realistic Sun-Earth-Moon system."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # Actual masses in solar masses
    m_sun = 1.0
    m_earth = 3.003e-6  # Earth mass in solar masses
    m_moon = 3.694e-8   # Moon mass in solar masses
    
    sim.add(m=m_sun)  # Sun at origin
    sim.add(m=m_earth, a=1.0, e=0.0167)  # Earth at 1 AU
    
    # Moon orbiting Earth (simplified as circular)
    moon_distance = 0.00257  # AU (384,400 km)
    moon_velocity = 0.213  # AU/yr (1.022 km/s)
    sim.add(m=m_moon, x=1.0 + moon_distance, y=0, z=0,
            vx=0, vy=2*np.pi + moon_velocity, vz=0)
    
    sim.move_to_com()
    return sim


def create_kozai_system(m_central=1.0, m_inner=0.001, m_perturber=0.5,
                       a_inner=1.0, a_perturber=5.0, i_perturber=65):
    """Create Kozai-Lidov system: inner body with inclined outer perturber."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    sim.add(m=m_central)  # Central star
    sim.add(m=m_inner, a=a_inner, e=0.1)  # Inner body
    
    # Outer perturber with high inclination (Kozai mechanism)
    i_rad = np.radians(i_perturber)
    sim.add(m=m_perturber, a=a_perturber, e=0.1, inc=i_rad)
    
    sim.move_to_com()
    return sim


def create_trojan_asteroids(m_sun=1.0, m_jupiter=0.001, n_trojans=5):
    """Create Sun-Jupiter system with Trojan asteroids at L4/L5 points."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # Sun and Jupiter
    sim.add(m=m_sun)
    sim.add(m=m_jupiter, a=5.2, e=0.05)  # Jupiter at 5.2 AU
    
    # Add Trojans near L4 point (60 degrees ahead of Jupiter)
    for i in range(n_trojans):
        # Small perturbations around L4
        angle = np.radians(60 + np.random.uniform(-5, 5))
        r = 5.2 + np.random.uniform(-0.1, 0.1)
        sim.add(m=0, a=r, f=angle)  # Massless test particles
    
    sim.move_to_com()
    return sim


def create_stable_lagrange(m_primary=1.0, m_secondary=0.3, separation=5.0):
    """Create stable L4/L5 configuration (like Jupiter's Trojans)."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    
    # Two main bodies in circular orbit
    sim.add(m=m_primary)  # e.g., Sun
    sim.add(m=m_secondary, a=separation)  # e.g., Jupiter
    
    # Add test particle at L4 (60 degrees ahead)
    # This is MUCH more stable than equal masses
    l4_angle = np.radians(60)
    sim.add(m=0, a=separation, f=l4_angle)  # Massless at L4
    
    # Add test particle at L5 (60 degrees behind)
    l5_angle = np.radians(-60)
    sim.add(m=0, a=separation, f=l5_angle)  # Massless at L5
    
    sim.move_to_com()
    return sim


def create_broucke_orbit():
    """Broucke's stable periodic orbit (more stable than figure-8)."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.integrator = "ias15"
    
    # Broucke A-15 orbit (stable periodic solution)
    sim.add(m=1.0, x=0.336130, y=0, z=0, 
            vx=0, vy=1.067877, vz=0)
    sim.add(m=1.0, x=0.768469, y=0, z=0, 
            vx=0, vy=-0.662036, vz=0)
    sim.add(m=1.0, x=-1.104599, y=0, z=0, 
            vx=0, vy=-0.405841, vz=0)
    
    sim.move_to_com()
    return sim


def simulate_and_extract_data(sim, t_max=100, n_points=1001):
    """
    Simulate the system and extract positions and forces.
    """
    times = np.linspace(0, t_max, n_points)
    n = len(sim.particles)
    positions = []
    forces = []

    for t in times:
        sim.integrate(t) # what does this do exactly? '.step(unit_time)' is not used, '.integrate(t)' is, why?
        
        # Extract positions
        pos = np.array([[p.x, p.y, p.z] for p in sim.particles])
        positions.append(pos)
        
        # Calculate forces
        force = np.zeros((n, 3))
        G = sim.G
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 0:
                        force[i] += G * sim.particles[i].m * sim.particles[j].m * r_vec / r_mag**3
        forces.append(force)
    
    return times, np.array(positions), np.array(forces)

