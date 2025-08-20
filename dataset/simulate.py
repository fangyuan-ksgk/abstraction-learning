# Simulate n-body system for various patterns & save GIFs 
# --------------------------------------------------------------------------------------------------------------------------

from nbody import * 
import numpy as np

# Dictionary of all available patterns
pattern_functions = {
    "random": create_random_n_body,
    "circular": create_circular_binary,
    "hierarchical": create_hierarchical_triple,
    "figure8": create_figure8_3body,
    "lagrange": create_lagrange_triangle,
    "pythagorean": create_pythagorean_3body,
    "sitnikov": create_sitnikov_problem,
    "sun_earth_moon": create_sun_earth_moon,
    "kozai": create_kozai_system,
    "trojan": create_trojan_asteroids
}

# Configuration for each pattern
pattern_configs = {
    "random": {
        "params": {"n": 3, "seed": 42, "com_center": True},
        "t_max": 20,
        "n_points": 201,
        "labels": ["Alpha", "Beta", "Gamma"],
        "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
        "trail_steps": 30
    },
    "circular": {
        "params": {"m1": 1.0, "m2": 0.5, "separation": 2.0, "com_center": True},
        "t_max": 10,
        "n_points": 101,
        "labels": ["Primary", "Secondary"],
        "colors": ["#FFD700", "#C0C0C0"],
        "trail_steps": 40
    },
    "hierarchical": {
        "params": {"m_central": 1.0, "m_inner": 0.01, "m_outer": 0.005},
        "t_max": 50,
        "n_points": 501,
        "labels": ["Star", "Inner Planet", "Outer Planet"],
        "colors": ["#FDB813", "#8B4513", "#4169E1"],
        "trail_steps": 50
    },
    "figure8": {
        "params": {},
        "t_max": 6.3259,  # One complete period
        "n_points": 201,
        "labels": ["Body A", "Body B", "Body C"],
        "colors": ["#FF1493", "#00CED1", "#FFD700"],
        "trail_steps": 60
    },
    "lagrange": {
        "params": {"m1": 1.0, "m2": 1.0, "m3": 1.0, "radius": 1.5},
        "t_max": 20,
        "n_points": 201,
        "labels": ["Star A", "Star B", "Star C"],
        "colors": ["#FF4500", "#32CD32", "#1E90FF"],
        "trail_steps": 40
    },
    "pythagorean": {
        "params": {},
        "t_max": 15,
        "n_points": 301,
        "labels": ["Mass 3", "Mass 4", "Mass 5"],
        "colors": ["#FF69B4", "#00FA9A", "#FFB347"],
        "trail_steps": 50
    },
    "sun_earth_moon": {
        "params": {},
        "t_max": 1,  # 1 year
        "n_points": 365,
        "labels": ["Sun", "Earth", "Moon"],
        "colors": ["#FDB813", "#006994", "#C0C0C0"],
        "trail_steps": 30
    }
}

def simulate_pattern(pattern_name, custom_config=None):
    """Simulate a specific pattern and create GIF."""
    
    if pattern_name not in pattern_functions:
        print(f"Error: Pattern '{pattern_name}' not found!")
        print(f"Available patterns: {list(pattern_functions.keys())}")
        return
    
    # Get configuration
    config = pattern_configs.get(pattern_name, {})
    if custom_config:
        config.update(custom_config)
    
    # Create simulation
    print(f"\n{'='*60}")
    print(f"Simulating {pattern_name.upper()} pattern...")
    print(f"{'='*60}")
    
    create_fn = pattern_functions[pattern_name]
    sim = create_fn(**config.get("params", {}))
    
    # Run simulation
    t_max = config.get("t_max", 10)
    n_points = config.get("n_points", 101)
    
    print(f"Simulating for t_max={t_max} with {n_points} points...")
    raw_data = simulate_and_extract_data(sim, t_max=t_max, n_points=n_points)
    
    ts, positions, forces = raw_data
    masses = np.array([p.m for p in sim.particles])
    
    # Determine labels and colors
    n_bodies = len(sim.particles)
    labels = config.get("labels", [f"Body {i+1}" for i in range(n_bodies)])[:n_bodies]
    colors = config.get("colors", [f"C{i}" for i in range(n_bodies)])[:n_bodies]
    
    # Create GIF
    save_path = f"{pattern_name}_nbody.gif"
    print(f"Creating GIF: {save_path}")
    
    create_nbody_gif(
        ts, positions, forces, masses,
        labels=labels,
        body_colors=colors,
        figsize=(10, 8),
        legend_scale=1.4,
        trail_steps=config.get("trail_steps", 30),
        force_scale=config.get("force_scale", 0.01),
        save_path=save_path,
        title=f"{pattern_name.replace('_', ' ').title()} System",
        fps=30
    )
    
    print(f"✓ Saved to {save_path}")
    return sim, (ts, positions, forces)

def simulate_all_patterns():
    """Generate GIFs for all available patterns."""
    results = {}
    for pattern_name in pattern_functions.keys():
        try:
            sim, data = simulate_pattern(pattern_name)
            results[pattern_name] = (sim, data)
        except Exception as e:
            print(f"✗ Failed to simulate {pattern_name}: {e}")
    return results

# Main execution
if __name__ == "__main__":
    results = simulate_all_patterns()
    