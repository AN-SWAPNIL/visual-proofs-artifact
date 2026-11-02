"""
Randomization Configuration for GRAPHTRAILS Dataset Generation

This module provides enhanced randomization capabilities to make the dataset
more unpredictable and reduce visual pattern bias in model training.

Key Improvements:
1. Wider node count ranges per property
2. Randomized visual appearance (node size, edge width, layout scale)
3. Layout perturbation (small noise to node positions)
4. Per-image unique seeding
5. Diverse graph generation algorithms
"""

import random
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================================
# RANDOMIZATION CONFIGURATION
# ============================================================================

@dataclass
class RandomizationConfig:
    """
    Configuration class for dataset randomization.
    All ranges are [min, max] inclusive.
    """
    
    # =========================================================================
    # NODE COUNT RANGES (per property for appropriate complexity)
    # =========================================================================
    NODE_RANGES: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        # Property: (min_nodes, max_nodes)
        'cut_vertex':      (12, 55),   # Needs enough nodes for clear clusters
        'connectivity':    (15, 60),   # Can handle larger graphs
        'bipartite':       (12, 50),   # Matrix size affects visibility
        'planarity':       (15, 55),   # Balance between complexity and layout
        'outer_planarity': (12, 50),   # Circular layout needs room
        'hamiltonian':     (12, 45),   # Cycle finding gets expensive
        'k_colorability':  (15, 50),   # Matrix representation
        'completeness':    (15, 45),   # Matrix gets dense quickly
    })
    
    # Default range for unknown properties
    DEFAULT_NODE_RANGE: Tuple[int, int] = (15, 50)
    
    # =========================================================================
    # VISUAL APPEARANCE RANDOMIZATION
    # =========================================================================
    
    # Node radius range (in layout coordinate units)
    NODE_RADIUS_MIN: float = 0.04
    NODE_RADIUS_MAX: float = 0.12
    
    # Edge width range (in points)
    EDGE_WIDTH_MIN: float = 0.6
    EDGE_WIDTH_MAX: float = 2.2
    
    # Layout scale factor range (multiplied to all positions)
    LAYOUT_SCALE_MIN: float = 0.85
    LAYOUT_SCALE_MAX: float = 1.15
    
    # Position noise scale (fraction of layout bound to add as noise)
    POSITION_NOISE_MIN: float = 0.0
    POSITION_NOISE_MAX: float = 0.03
    
    # Rotation angle range (degrees, applied to entire layout)
    ROTATION_MIN: float = 0.0
    ROTATION_MAX: float = 360.0
    
    # =========================================================================
    # GRAPH GENERATION DIVERSITY
    # =========================================================================
    
    # Edge probability variance (added to base probability)
    EDGE_PROB_VARIANCE: float = 0.15
    
    # Minimum/maximum edge probability bounds
    EDGE_PROB_MIN: float = 0.10
    EDGE_PROB_MAX: float = 0.65
    
    # =========================================================================
    # RANDOM SEED BEHAVIOR
    # =========================================================================
    
    # If True, each image gets a unique seed based on index
    USE_UNIQUE_SEEDS: bool = True
    
    # Base seed (combined with image index for unique seeds)
    BASE_SEED: int = 42


# Global configuration instance
CONFIG = RandomizationConfig()


# ============================================================================
# RANDOMIZATION FUNCTIONS
# ============================================================================

def get_random_node_range(property_name: str) -> Tuple[int, int]:
    """
    Get the randomized node count range for a property.
    
    Args:
        property_name: Name of the graph property
        
    Returns:
        (min_nodes, max_nodes) tuple
    """
    return CONFIG.NODE_RANGES.get(property_name, CONFIG.DEFAULT_NODE_RANGE)


def sample_node_count(property_name: str) -> int:
    """
    Sample a random node count for a property.
    
    Args:
        property_name: Name of the graph property
        
    Returns:
        Random node count within the property's range
    """
    min_n, max_n = get_random_node_range(property_name)
    return random.randint(min_n, max_n)


def sample_visual_params() -> Dict[str, float]:
    """
    Sample randomized visual parameters for a single image.
    
    Returns:
        Dict with keys: 'node_radius', 'edge_width', 'layout_scale', 
                       'position_noise', 'rotation_angle'
    """
    return {
        'node_radius': random.uniform(CONFIG.NODE_RADIUS_MIN, CONFIG.NODE_RADIUS_MAX),
        'edge_width': random.uniform(CONFIG.EDGE_WIDTH_MIN, CONFIG.EDGE_WIDTH_MAX),
        'layout_scale': random.uniform(CONFIG.LAYOUT_SCALE_MIN, CONFIG.LAYOUT_SCALE_MAX),
        'position_noise': random.uniform(CONFIG.POSITION_NOISE_MIN, CONFIG.POSITION_NOISE_MAX),
        'rotation_angle': random.uniform(CONFIG.ROTATION_MIN, CONFIG.ROTATION_MAX),
    }


def sample_edge_probability(base_prob: float = 0.3) -> float:
    """
    Sample a randomized edge probability around a base value.
    
    Args:
        base_prob: Base probability to vary around
        
    Returns:
        Clamped random probability
    """
    variance = random.uniform(-CONFIG.EDGE_PROB_VARIANCE, CONFIG.EDGE_PROB_VARIANCE)
    prob = base_prob + variance
    return max(CONFIG.EDGE_PROB_MIN, min(CONFIG.EDGE_PROB_MAX, prob))


def apply_layout_randomization(
    pos: Dict[int, Tuple[float, float]],
    visual_params: Optional[Dict[str, float]] = None
) -> Dict[int, Tuple[float, float]]:
    """
    Apply randomization transformations to node positions.
    
    Transformations applied (in order):
    1. Scale (uniform scaling of positions)
    2. Rotation (rotate entire layout)
    3. Position noise (small random perturbation)
    
    Args:
        pos: Dict mapping node -> (x, y) position
        visual_params: Optional dict from sample_visual_params()
        
    Returns:
        New position dict with transformations applied
    """
    if visual_params is None:
        visual_params = sample_visual_params()
    
    scale = visual_params.get('layout_scale', 1.0)
    noise_scale = visual_params.get('position_noise', 0.0)
    rotation_deg = visual_params.get('rotation_angle', 0.0)
    
    # Convert rotation to radians
    rotation_rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
    
    # Calculate layout bounds for noise scaling
    if pos:
        max_coord = max(max(abs(x), abs(y)) for x, y in pos.values())
    else:
        max_coord = 1.0
    
    new_pos = {}
    for node, (x, y) in pos.items():
        # 1. Scale
        x_scaled = x * scale
        y_scaled = y * scale
        
        # 2. Rotate
        x_rot = x_scaled * cos_r - y_scaled * sin_r
        y_rot = x_scaled * sin_r + y_scaled * cos_r
        
        # 3. Add noise
        if noise_scale > 0:
            noise_x = random.uniform(-1, 1) * noise_scale * max_coord
            noise_y = random.uniform(-1, 1) * noise_scale * max_coord
            x_final = x_rot + noise_x
            y_final = y_rot + noise_y
        else:
            x_final = x_rot
            y_final = y_rot
        
        new_pos[node] = (x_final, y_final)
    
    return new_pos


def set_image_seed(image_index: int, base_seed: Optional[int] = None):
    """
    Set random seed for a specific image to ensure variety.
    
    Args:
        image_index: Index of the current image being generated
        base_seed: Base seed (uses CONFIG.BASE_SEED if None)
    """
    if base_seed is None:
        base_seed = CONFIG.BASE_SEED
    
    if CONFIG.USE_UNIQUE_SEEDS:
        # Combine base seed with image index for uniqueness
        unique_seed = base_seed * 10000 + image_index
    else:
        unique_seed = base_seed
    
    random.seed(unique_seed)
    np.random.seed(unique_seed)


# ============================================================================
# GRAPH TYPE SELECTION
# ============================================================================

# Available graph generation algorithms per property
GRAPH_TYPES = {
    'connected': [
        'erdos_renyi',
        'watts_strogatz', 
        'barabasi_albert',
        'random_regular',
        'powerlaw_cluster',
        'random_geometric',
        'newman_watts_strogatz',
    ],
    'planar': [
        'tree',
        'maximal_planar',
        'triangulation_insert',
        'random_edge_add',
        'grid',
        'wheel',
        'delaunay_like',
    ],
    'bipartite': [
        'random_bipartite',
        'balanced_bipartite',
        'star_bipartite',
        'complete_bipartite_minus',
        'path_bipartite',
    ],
}


def sample_graph_type(category: str) -> str:
    """
    Sample a random graph generation algorithm from a category.
    
    Args:
        category: One of 'connected', 'planar', 'bipartite'
        
    Returns:
        Name of the selected algorithm
    """
    types = GRAPH_TYPES.get(category, GRAPH_TYPES['connected'])
    return random.choice(types)


# ============================================================================
# PROPERTY-SPECIFIC RANDOMIZATION
# ============================================================================

def get_property_edge_prob(property_name: str) -> float:
    """
    Get base edge probability for a property's graph generation.
    
    Different properties may benefit from different densities.
    """
    base_probs = {
        'cut_vertex': 0.30,
        'connectivity': 0.35,
        'bipartite': 0.35,
        'planarity': 0.25,
        'outer_planarity': 0.25,
        'hamiltonian': 0.35,
        'k_colorability': 0.30,
        'completeness': 0.50,
    }
    base = base_probs.get(property_name, 0.30)
    return sample_edge_probability(base)


# ============================================================================
# CANVAS PLACEMENT — Unified for both graph and matrix visualizations
# ============================================================================

def sample_canvas_placement() -> Dict[str, float]:
    """
    Sample random placement of content on the 400x400 canvas.
    
    Modes (randomly chosen):
      full:     content fills 88-100% of canvas  (weighted 2x)
      small:    content fills 45-72%, floating in whitespace
      wide:     landscape stress — wider than tall
      tall:     portrait stress — taller than wide
      diagonal: uniform scale + rotation 10-18 degrees
    
    Content is NEVER cropped — all information stays within canvas bounds.
    """
    mode = random.choice(['full', 'full', 'small', 'wide', 'tall', 'diagonal'])

    if mode == 'full':
        w = random.uniform(0.88, 1.0)
        h = random.uniform(0.88, 1.0)
        rot = random.uniform(-3, 3)
    elif mode == 'small':
        base = random.uniform(0.45, 0.72)
        w = base * random.uniform(0.92, 1.08)
        h = base * random.uniform(0.92, 1.08)
        rot = random.uniform(-8, 8)
    elif mode == 'wide':
        w = random.uniform(0.78, 1.0)
        h = random.uniform(0.42, 0.68)
        rot = random.uniform(-4, 4)
    elif mode == 'tall':
        w = random.uniform(0.42, 0.68)
        h = random.uniform(0.78, 1.0)
        rot = random.uniform(-4, 4)
    else:  # diagonal
        base = random.uniform(0.55, 0.82)
        w = base
        h = base
        rot = random.choice([-1, 1]) * random.uniform(10, 18)

    w = min(1.0, max(0.4, w))
    h = min(1.0, max(0.4, h))

    margin_x = (1.0 - w) / 2.0
    margin_y = (1.0 - h) / 2.0
    offset_x = random.uniform(-margin_x, margin_x) if margin_x > 0.01 else 0.0
    offset_y = random.uniform(-margin_y, margin_y) if margin_y > 0.01 else 0.0

    return {
        'content_w': w,
        'content_h': h,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'rotation': rot,
        'mode': mode,
    }


def sample_matrix_visual_params() -> Dict[str, float]:
    """
    Sample visual parameters for matrix visualizations.
    Combines canvas placement with matrix-specific line width randomization.
    """
    placement = sample_canvas_placement()
    placement['grid_line_width'] = random.uniform(0.2, 1.5)
    placement['block_line_width'] = random.uniform(1.0, 3.0)
    placement['border_line_width'] = random.uniform(0.5, 2.5)
    return placement


# ============================================================================
# COLOR SCHEME RANDOMIZATION
# ============================================================================

_NODE_COLOR_PAIRS = [
    ('#4ECDC4', '#2EA39C'),  # Teal (original)
    ('#5B9BD5', '#2F75B5'),  # Ocean blue
    ('#66BB6A', '#43A047'),  # Forest green
    ('#FF8A65', '#E64A19'),  # Coral
    ('#AB47BC', '#7B1FA2'),  # Purple
    ('#FFB74D', '#F57C00'),  # Gold/amber
    ('#78909C', '#546E7A'),  # Slate
    ('#EC407A', '#C2185B'),  # Pink
    ('#26C6DA', '#00ACC1'),  # Cyan
    ('#8D6E63', '#5D4037'),  # Brown
]

_HIGHLIGHT_COLORS = ['#FF6B6B', '#EF5350', '#FF7043', '#E53935', '#D32F2F']
_EDGE_COLORS = ['#666666', '#777777', '#888888', '#555555', '#7F8C8D']
_BORDER_COLORS = ['#333333', '#2C3E50', '#37474F', '#424242', '#263238']
_BACKGROUND_COLORS = [
    '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF',
    '#F5F5DC', '#F0F8FF', '#FFF8E7', '#F5F5F5',
    '#FFFDE7', '#F3E5F5', '#E8F5E9', '#E3F2FD',
]
_SHADE_COLORS = ['#999999', '#888888', '#AAAAAA', '#7F8C8D']


def sample_color_scheme() -> Optional[Dict[str, str]]:
    """
    Sample a random color scheme for graph/matrix visualization.
    Returns None ~40% of the time (use default COLORS).
    """
    if random.random() < 0.4:
        return None

    node_reg, node_cycle = random.choice(_NODE_COLOR_PAIRS)
    highlight = random.choice(_HIGHLIGHT_COLORS)

    return {
        'regular_node': node_reg,
        'cycle_cell': node_cycle,
        'highlight_node': highlight,
        'highlight_edge': highlight,
        'regular_edge': random.choice(_EDGE_COLORS),
        'node_border': random.choice(_BORDER_COLORS),
        'background': random.choice(_BACKGROUND_COLORS),
        'shade_color': random.choice(_SHADE_COLORS),
    }


def randomize_matrix_order(node_order: list, preserve_blocks: bool = False) -> list:
    """
    Optionally shuffle node ordering within blocks for matrix visualization.
    
    For YES cases (where blocks are meaningful), preserve_blocks=True keeps
    nodes within their blocks but shuffles order within each block.
    
    For NO cases, full shuffle adds more visual variety.
    
    Args:
        node_order: Original list of node indices
        preserve_blocks: If True, only shuffle within blocks
        
    Returns:
        Shuffled node order
    """
    if preserve_blocks:
        # Keep original order but could add intra-block shuffling
        # For now, just return as-is to preserve block structure
        return node_order
    else:
        # For NO cases, we can add slight randomization
        # But usually matrix order is important for visual proof
        return node_order


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def describe_config() -> str:
    """Return a string describing the current randomization configuration."""
    desc = [
        "GRAPHTRAILS Randomization Configuration",
        "=" * 50,
        "",
        "Node Count Ranges:",
    ]
    for prop, (min_n, max_n) in CONFIG.NODE_RANGES.items():
        desc.append(f"  {prop}: {min_n}-{max_n} nodes")
    
    desc.extend([
        "",
        "Visual Parameters:",
        f"  Node radius: {CONFIG.NODE_RADIUS_MIN:.2f} - {CONFIG.NODE_RADIUS_MAX:.2f}",
        f"  Edge width: {CONFIG.EDGE_WIDTH_MIN:.1f} - {CONFIG.EDGE_WIDTH_MAX:.1f}",
        f"  Layout scale: {CONFIG.LAYOUT_SCALE_MIN:.2f} - {CONFIG.LAYOUT_SCALE_MAX:.2f}",
        f"  Position noise: {CONFIG.POSITION_NOISE_MIN:.3f} - {CONFIG.POSITION_NOISE_MAX:.3f}",
        f"  Rotation: {CONFIG.ROTATION_MIN:.0f}° - {CONFIG.ROTATION_MAX:.0f}°",
        "",
        "Edge Probability:",
        f"  Variance: ±{CONFIG.EDGE_PROB_VARIANCE:.2f}",
        f"  Bounds: {CONFIG.EDGE_PROB_MIN:.2f} - {CONFIG.EDGE_PROB_MAX:.2f}",
        "",
        f"Unique seeds per image: {CONFIG.USE_UNIQUE_SEEDS}",
    ])
    
    return "\n".join(desc)


if __name__ == '__main__':
    # Test the randomization configuration
    print(describe_config())
    print("\n" + "=" * 50)
    print("\nSample visual params (3 examples):")
    for i in range(3):
        set_image_seed(i)
        params = sample_visual_params()
        print(f"\n  Image {i+1}:")
        for k, v in params.items():
            print(f"    {k}: {v:.4f}")
    
    print("\n\nSample node counts (per property):")
    set_image_seed(0)
    for prop in CONFIG.NODE_RANGES.keys():
        counts = [sample_node_count(prop) for _ in range(5)]
        print(f"  {prop}: {counts}")
