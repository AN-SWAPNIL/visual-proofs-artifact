"""
GRAPHTRAILS Dataset Generator for Graph Property Classification

Modular framework for generating training data for CNN/Vision GNN.
This is IMAGE CLASSIFICATION (not object detection) - no bounding box labels needed.

Following GRAPHTRAILS visual proof conventions for clear, interpretable visualizations.

Available Properties:
- Cut-Vertex: Graph has/doesn't have cut-vertex
- Connectivity: Graph is connected/disconnected
- Non-Bipartite: Graph has/doesn't have odd cycle
- Planarity: Graph is planar/non-planar
- Outer-Planarity: Graph is outer-planar/non-outer-planar
- Hamiltonian: Graph has/doesn't have Hamiltonian cycle
- k-Colorability: Graph is/isn't k-colorable (k=4 default)
- Completeness: Graph is/isn't complete

Usage:
    # Generate single property
    python graphtrails_dataset_generator.py --property cut_vertex --n_per_class 150
    python graphtrails_dataset_generator.py --property connectivity --n_per_class 150
    
    # Generate ALL properties at once (recommended)
    python graphtrails_dataset_generator.py --property all --n_per_class 150
    
    # This will generate 2,700 images total (9 properties × 300 images)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib.collections import LineCollection
import networkx as nx
import os
import random
import json
import argparse
from typing import List, Tuple, Set, Dict, Optional

# Import property-specific generators
from cut_vertex_generator import (
    generate_cut_vertex_graph_yes,
    generate_cut_vertex_graph_no,
    generate_cut_vertex_graph_s_no,
    generate_cut_vertex_graph_d_no,
    layout_cut_vertex_yes,
    layout_cut_vertex_no,
    layout_cut_vertex_s_no,
    layout_cut_vertex_d_no,
    get_cut_vertex_metadata
)

from connectivity_generator import (
    generate_connectivity_graph_yes,
    generate_connectivity_graph_no,
    generate_connectivity_graph_s_no,
    generate_connectivity_graph_d_no,
    layout_connectivity_yes,
    layout_connectivity_no,
    layout_connectivity_s_no,
    layout_connectivity_d_no,
    get_connectivity_metadata
)

from bipartite_generator import (
    generate_bipartite_graph_yes,
    generate_bipartite_graph_no,
    generate_bipartite_graph_d_no,
    generate_bipartite_graph_s_no,
    layout_bipartite_yes,
    layout_bipartite_no,
    get_bipartite_metadata
)

from planarity_generator import (
    generate_planarity_graph_yes,
    generate_planarity_graph_no,
    generate_planarity_graph_s_no,
    generate_planarity_graph_d_no,
    layout_planarity_yes,
    layout_planarity_no,
    get_planarity_metadata
)

from outer_planarity_generator import (
    generate_outer_planarity_graph_yes,
    generate_outer_planarity_graph_no,
    generate_outer_planarity_graph_s_no,
    generate_outer_planarity_graph_d_no,
    layout_outer_planarity_yes,
    layout_outer_planarity_no,
    get_outer_planarity_metadata
)

from hamiltonian_generator import (
    generate_hamiltonian_graph_yes,
    generate_hamiltonian_graph_no,
    generate_hamiltonian_graph_s_no,
    generate_hamiltonian_graph_d_no,
    layout_hamiltonian_yes,
    layout_hamiltonian_no,
    layout_hamiltonian_s_no,
    layout_hamiltonian_d_no,
    get_hamiltonian_metadata
)

from k_colorability_generator import (
    generate_k_colorability_graph_yes,
    generate_k_colorability_graph_no,
    generate_k_colorability_graph_s_no,
    generate_k_colorability_graph_d_no,
    layout_k_colorability_yes,
    layout_k_colorability_no,
    layout_k_colorability_d_no,
    get_k_colorability_metadata
)

from completeness_generator import (
    generate_completeness_graph_yes,
    generate_completeness_graph_no,
    generate_completeness_graph_s_no,
    generate_completeness_graph_d_no,
    layout_completeness_yes,
    layout_completeness_no,
    layout_completeness_d_no,
    get_completeness_metadata
)

# Import new randomization and reasoning modules
try:
    from randomization_config import (
        sample_visual_params, apply_layout_randomization,
        set_image_seed, CONFIG as RAND_CONFIG, describe_config,
        sample_matrix_visual_params, sample_canvas_placement,
        sample_color_scheme
    )
    USE_RANDOMIZATION = True
except ImportError:
    USE_RANDOMIZATION = False
    print("Warning: randomization_config not found, using default settings")

try:
    from reasoning_text_generator import generate_reasoning_text, get_answer_with_reasoning
    USE_REASONING = True
except ImportError:
    USE_REASONING = False
    print("Warning: reasoning_text_generator not found, reasoning texts disabled")


# Set random seed for reproducibility (will be overridden per-image if USE_RANDOMIZATION)
np.random.seed(42)
random.seed(42)


# ============================================================================
# COLOR SCHEME (from Plan.md - GRAPHTRAILS style)
# ============================================================================
COLORS = {
    'regular_node': '#4ECDC4',      # TEAL
    'cycle_cell': '#2EA39C',        # Slightly deeper teal for odd-cycle matrix cells
    'highlight_node': '#FF6B6B',    # RED
    'regular_edge': '#666666',      # GRAY
    'highlight_edge': '#FF6B6B',    # RED
    'node_border': '#333333',       # DARK GRAY
    'background': '#FFFFFF',        # WHITE
}

# ============================================================================
# IMAGE SPECIFICATIONS (with randomization support)
# ============================================================================
IMG_SIZE = 400
DPI = 150

# Base values (will be randomized per-image when USE_RANDOMIZATION=True)
NODE_RADIUS_BASE = 0.06
EDGE_WIDTH_BASE = 1.2
FONT_SIZE = 7
LAYOUT_BOUND = 3.5

# For backward compatibility, keep fixed defaults
NODE_RADIUS = NODE_RADIUS_BASE
EDGE_WIDTH = EDGE_WIDTH_BASE


# ============================================================================
# PROPERTY CONFIGURATIONS
# ============================================================================

PROPERTY_CONFIGS = {
    'cut_vertex': {
        'generate_yes': generate_cut_vertex_graph_yes,
        'generate_no': generate_cut_vertex_graph_no,  # Unified NO generator
        'generate_s_no': generate_cut_vertex_graph_s_no,
        'generate_d_no': generate_cut_vertex_graph_d_no,
        'layout_yes': layout_cut_vertex_yes,
        'layout_s_no': layout_cut_vertex_s_no,
        'layout_d_no': layout_cut_vertex_d_no,
        'get_metadata': get_cut_vertex_metadata,
        'description': 'Cut-Vertex Detection',
        'has_auxiliary_data': True,  # Returns (G, cut_vertices)
        's_no_has_auxiliary': True,  # s_no returns (G, layout_info)
        'd_no_has_auxiliary': True,  # d_no returns (G, ears)
        'yes_bold_nodes': True,      # YES class uses bold nodes for cut-vertex
        's_no_info_key': 'layout_info',  # Key in info dict for s_no auxiliary
        'd_no_info_key': 'ears'          # Key in info dict for d_no auxiliary
    },
    'connectivity': {
        'generate_yes': generate_connectivity_graph_yes,
        'generate_no': generate_connectivity_graph_no,  # Unified NO generator
        'generate_s_no': generate_connectivity_graph_s_no,
        'generate_d_no': generate_connectivity_graph_d_no,
        'layout_yes': layout_connectivity_yes,
        'layout_s_no': layout_connectivity_s_no,
        'layout_d_no': layout_connectivity_d_no,
        'get_metadata': get_connectivity_metadata,
        'description': 'Graph Connectivity (GRAPHTRAILS Fig 3c)',
        'has_auxiliary_data': True,
        's_no_has_auxiliary': True,   # s_no returns (G, layout_info)
        'd_no_has_auxiliary': True,   # d_no returns (G, None) - tuple
        'yes_bold_nodes': False,
        'yes_bold_edges': True,       # YES class uses bold edges for spanning tree
        's_no_info_key': None,        # s_no uses entire info dict
        'd_no_info_key': None         # d_no uses None
    },
    'bipartite': {
        'generate_yes': generate_bipartite_graph_yes,
        'generate_no': generate_bipartite_graph_no,  # Unified NO generator
        'generate_s_no': generate_bipartite_graph_s_no,
        'generate_d_no': generate_bipartite_graph_d_no,
        'layout_yes': layout_bipartite_no,   # YES shows two partitions (matrix blocks)
        'layout_s_no': layout_bipartite_no,  # s_no uses same bipartite-like layout
        'layout_d_no': layout_bipartite_yes, # d_no uses odd-cycle visualization
        'get_metadata': get_bipartite_metadata,
        'description': 'Bipartiteness Detection (matrix two-block proof)',
        'has_auxiliary_data': True,
        's_no_has_auxiliary': True,
        'd_no_has_auxiliary': True,
        'yes_bold_nodes': False,
        's_no_info_key': 'pseudo_bipartite_sets',
        'd_no_info_key': 'odd_cycle'
    },

    'planarity': {
        'generate_yes': generate_planarity_graph_yes,
        'generate_no': generate_planarity_graph_no,  # Unified NO generator
        'generate_s_no': generate_planarity_graph_s_no,  # s_no: same layout but non-planar (crossing edges)
        'generate_d_no': generate_planarity_graph_d_no,  # d_no: different layout (original NO)
        'layout_yes': layout_planarity_yes,
        'layout_s_no': layout_planarity_yes,  # s_no uses same layout as YES
        'layout_d_no': layout_planarity_no,
        'get_metadata': get_planarity_metadata,
        'description': 'Planarity Detection (GRAPHTRAILS Fig 8)',
        'has_auxiliary_data': True,
        's_no_has_auxiliary': True,
        'd_no_has_auxiliary': True,
        'yes_bold_nodes': False,
        's_no_info_key': 'template_pos',
        'd_no_info_key': 'minor_info'
    },
    'outer_planarity': {
        'generate_yes': generate_outer_planarity_graph_yes,
        'generate_no': generate_outer_planarity_graph_no,  # Unified NO generator
        'generate_s_no': generate_outer_planarity_graph_s_no,  # s_no: same layout but non-outerplanar (crossing edges)
        'generate_d_no': generate_outer_planarity_graph_d_no,  # d_no: different layout existing implementation
        'layout_yes': layout_outer_planarity_yes,
        'layout_s_no': layout_outer_planarity_yes,  # s_no uses same layout as YES
        'layout_d_no': layout_outer_planarity_no,
        'get_metadata': get_outer_planarity_metadata,
        'description': 'Outer-Planarity Detection (GRAPHTRAILS Fig 8b/8d)',
        'has_auxiliary_data': True,
        's_no_has_auxiliary': True,
        'd_no_has_auxiliary': True,
        'yes_bold_nodes': False,
        's_no_info_key': 'template_pos',
        'd_no_info_key': 'minor_info'
    },
    'hamiltonian': {
        'generate_yes': generate_hamiltonian_graph_yes,
        'generate_no': generate_hamiltonian_graph_no,  # Unified NO generator
        'generate_s_no': generate_hamiltonian_graph_s_no,
        'generate_d_no': generate_hamiltonian_graph_d_no,
        'layout_yes': layout_hamiltonian_yes,
        'layout_s_no': layout_hamiltonian_s_no,
        'layout_d_no': layout_hamiltonian_d_no,
        'get_metadata': get_hamiltonian_metadata,
        'description': 'Hamiltonian Cycle Detection (GRAPHTRAILS Fig 4b)',
        'has_auxiliary_data': True,
        's_no_has_auxiliary': True,
        'd_no_has_auxiliary': True,
        'yes_bold_nodes': False,
        'yes_bold_edges': True,     # YES class uses bold edges for Hamiltonian cycle
        's_no_bold_edges': True,    # s_no also uses bold edges for longest cycle
        'd_no_bold_nodes': True,    # d_no uses bold nodes for cut-set S
        's_no_info_key': 'longest_cycle',
        'd_no_info_key': 'cut_set_data'
    },
    'k_colorability': {
        'generate_yes': generate_k_colorability_graph_yes,
        'generate_no': generate_k_colorability_graph_no,  # Unified NO generator
        'generate_s_no': generate_k_colorability_graph_s_no,
        'generate_d_no': generate_k_colorability_graph_d_no,
        'layout_yes': layout_k_colorability_yes,
        'layout_s_no': layout_k_colorability_no,  # s_no uses layout_no to show >4 blocks
        'layout_d_no': layout_k_colorability_d_no,   # d_no highlights 5-clique cells
        'get_metadata': get_k_colorability_metadata,
        'description': 'k-Colorability Detection (GRAPHTRAILS Fig 5)',
        'has_auxiliary_data': True,
        's_no_has_auxiliary': True,
        'd_no_has_auxiliary': True,
        'yes_bold_nodes': False,
        's_no_info_key': 'coloring',
        'd_no_info_key': 'clique_nodes'
    },

    'completeness': {
        'generate_yes': generate_completeness_graph_yes,
        'generate_no': generate_completeness_graph_no,  # Unified NO generator
        'generate_s_no': generate_completeness_graph_s_no,  # s_no uses unified NO generator
        'generate_d_no': generate_completeness_graph_d_no,  # d_no uses unified NO generator
        'layout_yes': layout_completeness_yes,
        'layout_s_no': layout_completeness_no,  # SAME layout as YES
        'layout_d_no': layout_completeness_d_no,  # DIFFERENT layout: shade empty cells
        'get_metadata': get_completeness_metadata,
        'description': 'Completeness Detection (GRAPHTRAILS Fig 7)',
        'has_auxiliary_data': True,
        's_no_has_auxiliary': True,
        'd_no_has_auxiliary': True,
        'yes_bold_nodes': False,
        's_no_info_key': 'missing_edge',
        'd_no_info_key': 'all_missing_edges'
    }, 
}


# ============================================================================
# UNIVERSAL VISUALIZATION FUNCTION
# ============================================================================

def visualize_graph(
    G: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    output_path: str,
    highlight_edges: Optional[Set[Tuple[int, int]]] = None,
    node_colors: Optional[Dict[int, str]] = None,
    bold_nodes: Optional[Set[int]] = None,
    bold_edges: Optional[Set[Tuple[int, int]]] = None,
    visual_params: Optional[Dict[str, float]] = None,
    color_scheme: Optional[Dict[str, str]] = None
):
    """
    Universal graph rendering function for all properties.
    
    Canvas placement: the graph is drawn at a random size/position/aspect
    on the 400x400 background. Content is NEVER cropped.
    """
    # Merge color scheme with defaults
    colors = {**COLORS, **(color_scheme or {})}

    # Apply visual randomization if parameters provided
    if visual_params and USE_RANDOMIZATION:
        pos = apply_layout_randomization(pos, visual_params)
        current_node_radius = visual_params.get('node_radius', NODE_RADIUS_BASE)
        current_edge_width = visual_params.get('edge_width', EDGE_WIDTH_BASE)
    else:
        current_node_radius = NODE_RADIUS_BASE
        current_edge_width = EDGE_WIDTH_BASE
    
    # Create figure with exact pixel size
    fig_size = IMG_SIZE / DPI
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=DPI)
    
    # Make axes fill the entire figure (no matplotlib padding)
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])
    
    # Compute natural content bounds from node positions
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        max_x = max(abs(min(all_x)), abs(max(all_x)))
        max_y = max(abs(min(all_y)), abs(max(all_y)))
        bound = max(LAYOUT_BOUND, max_x + 1.0, max_y + 1.0)
    else:
        bound = LAYOUT_BOUND
    
    # Canvas placement: random size, position, and aspect ratio
    if USE_RANDOMIZATION:
        placement = sample_canvas_placement()
        cw = placement['content_w']
        ch = placement['content_h']
        
        # View half-ranges: larger range = more whitespace = smaller content
        view_half_x = bound / cw
        view_half_y = bound / ch
        
        # Offset: convert canvas-fraction offset to data coordinates
        shift_x = placement['offset_x'] * 2 * view_half_x
        shift_y = placement['offset_y'] * 2 * view_half_y
        
        ax.set_xlim(shift_x - view_half_x, shift_x + view_half_x)
        ax.set_ylim(shift_y - view_half_y, shift_y + view_half_y)
        
        # Node radius scaled to average view extent
        avg_view = (view_half_x + view_half_y) / 2.0
        scaled_node_radius = current_node_radius * (avg_view / LAYOUT_BOUND)
    else:
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        scaled_node_radius = current_node_radius * (bound / LAYOUT_BOUND)
    
    ax.axis('off')
    
    # Normalize highlight_edges to set of tuples (both directions)
    highlighted = set()
    if highlight_edges:
        for u, v in highlight_edges:
            highlighted.add((u, v))
            highlighted.add((v, u))
    
    # Normalize bold_edges to set of tuples (both directions)
    bolded_edges = set()
    if bold_edges:
        for u, v in bold_edges:
            bolded_edges.add((u, v))
            bolded_edges.add((v, u))
    
    # Draw edges in three passes: regular, bold, highlighted (on top)
    for u, v in G.edges():
        if (u, v) not in highlighted and (v, u) not in highlighted and \
           (u, v) not in bolded_edges and (v, u) not in bolded_edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], 
                    color=colors['regular_edge'], 
                    linewidth=current_edge_width, 
                    alpha=0.6, zorder=1)
    
    for u, v in G.edges():
        if ((u, v) in bolded_edges or (v, u) in bolded_edges) and \
           (u, v) not in highlighted and (v, u) not in highlighted:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], 
                    color=colors['regular_edge'], 
                    linewidth=current_edge_width * 2.5,
                    alpha=0.9, zorder=2)
    
    for u, v in G.edges():
        if (u, v) in highlighted or (v, u) in highlighted:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], 
                    color=colors['highlight_edge'], 
                    linewidth=current_edge_width * 1.5, 
                    alpha=0.9, zorder=3)
    
    # Draw nodes
    bold_set = set(bold_nodes) if bold_nodes else set()
    
    for node in G.nodes():
        x, y = pos[node]
        
        if node_colors and node in node_colors:
            node_color = node_colors[node]
        else:
            node_color = colors['regular_node']
        
        border_width = 2.5 if node in bold_set else 0.8
        
        circle = Circle((x, y), radius=scaled_node_radius, 
                        facecolor=node_color,
                        edgecolor=colors['node_border'],
                        linewidth=border_width,
                        zorder=10)
        ax.add_patch(circle)
    
    fig.savefig(output_path, dpi=DPI,
                facecolor=colors['background'], edgecolor='none')
    plt.close(fig)


def visualize_matrix(
    matrix_data: Dict,
    output_path: str,
    color_scheme: Optional[Dict[str, str]] = None
):
    """
    Visualize graph as adjacency matrix representation.
    
    Canvas placement: the matrix is drawn at a random size/position/aspect
    on the 400x400 background. Non-square cells create stress effects.
    Content is NEVER cropped.
    """
    # Merge color scheme with defaults
    colors = {**COLORS, **(color_scheme or {})}

    matrix = matrix_data['matrix']
    highlight_cells = matrix_data.get('highlight_cells', [])
    block_boundaries = matrix_data.get('block_boundaries', [])
    
    n = matrix.shape[0]
    
    # Sample matrix visual randomization (includes canvas placement)
    if USE_RANDOMIZATION:
        mvp = sample_matrix_visual_params()
        mat_w = mvp['content_w']
        mat_h = mvp['content_h']
        off_x = mvp['offset_x']
        off_y = mvp['offset_y']
        grid_lw = mvp['grid_line_width']
        block_lw = mvp['block_line_width']
        border_lw = mvp['border_line_width']
        rot_angle = mvp['rotation']
    else:
        mat_w = mat_h = 1.0
        off_x = off_y = 0.0
        grid_lw = 0.8
        block_lw = 1.5
        border_lw = 1.0
        rot_angle = 0.0
    
    # Create figure with exact pixel size
    fig_size = IMG_SIZE / DPI
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=DPI)
    
    # Make axes fill the entire figure
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor(colors['background'])
    fig.patch.set_facecolor(colors['background'])
    
    # Ensure the rotated bounding box fits within the [0,1] canvas.
    # A rect of size mat_w × mat_h rotated by θ has bounding box:
    #   bb_w = mat_w·|cos θ| + mat_h·|sin θ|
    #   bb_h = mat_w·|sin θ| + mat_h·|cos θ|
    # Shrink mat_w/mat_h so bb fits, then clamp the center.
    if abs(rot_angle) > 0.1:
        theta = abs(np.radians(rot_angle))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        bb_w = mat_w * cos_t + mat_h * sin_t
        bb_h = mat_w * sin_t + mat_h * cos_t
        if bb_w > 1.0 or bb_h > 1.0:
            scale = min(1.0 / bb_w, 1.0 / bb_h) * 0.97  # 3% safety margin
            mat_w *= scale
            mat_h *= scale
            bb_w *= scale
            bb_h *= scale
            off_x *= scale
            off_y *= scale
        # Center must sit far enough from edges for the rotated half-extents
        half_bb_w = bb_w / 2.0
        half_bb_h = bb_h / 2.0
        cx = np.clip(0.5 + off_x, half_bb_w, 1.0 - half_bb_w)
        cy = np.clip(0.5 + off_y, half_bb_h, 1.0 - half_bb_h)
        x0 = cx - mat_w / 2.0
        y0 = cy - mat_h / 2.0
    else:
        # Matrix drawing region: (x0, y0) is bottom-left corner
        margin_x = (1.0 - mat_w) / 2.0
        margin_y = (1.0 - mat_h) / 2.0
        x0 = margin_x + off_x
        y0 = margin_y + off_y
        # Clamp so matrix stays within [0,1] canvas
        x0 = max(0.0, min(x0, 1.0 - mat_w))
        y0 = max(0.0, min(y0, 1.0 - mat_h))
    
    # Cell sizes — non-square when mat_w != mat_h (stress effect)
    cell_w = mat_w / n
    cell_h = mat_h / n
    
    # Apply rotation via transform (rotate around matrix center)
    if abs(rot_angle) > 0.1:
        import matplotlib.transforms as mtransforms
        cx = x0 + mat_w / 2.0
        cy = y0 + mat_h / 2.0
        t = mtransforms.Affine2D().rotate_deg_around(cx, cy, rot_angle) + ax.transData
    else:
        t = ax.transData
    
    # Draw shaded missing cells (d_no completeness)
    shade_missing_cells = matrix_data.get('shade_missing_cells', [])
    shade_set = set(shade_missing_cells)
    shade_color = colors.get('shade_color', '#999999')

    for (i, j) in shade_set:
        if i == j:
            continue
        x = x0 + j * cell_w
        y = y0 + (n - 1 - i) * cell_h
        rect = plt.Rectangle((x, y), cell_w, cell_h, facecolor=shade_color,
                             edgecolor='none', alpha=1.0, transform=t)
        ax.add_patch(rect)

    # Draw matrix cells (present edges)
    highlight_set = set(highlight_cells)

    for i in range(n):
        for j in range(n):
            x = x0 + j * cell_w
            y = y0 + (n - 1 - i) * cell_h

            if matrix[i, j] == 1:
                if (i, j) in highlight_set:
                    color = colors.get('cycle_cell', '#2EA39C')
                else:
                    color = colors['regular_node']

                rect = plt.Rectangle((x, y), cell_w, cell_h,
                                     facecolor=color, edgecolor='none', transform=t)
                ax.add_patch(rect)
    
    # Grid lines and block boundaries
    is_grid_matrix = len(block_boundaries) > 8
    
    if is_grid_matrix:
        # Completeness style: black grid lines
        for i in range(n + 1):
            ax.plot([x0, x0 + mat_w], [y0 + i * cell_h, y0 + i * cell_h],
                    color='#000000', linewidth=grid_lw, transform=t)
            ax.plot([x0 + i * cell_w, x0 + i * cell_w], [y0, y0 + mat_h],
                    color='#000000', linewidth=grid_lw, transform=t)
    else:
        # Light grid lines
        for i in range(n + 1):
            ax.plot([x0, x0 + mat_w], [y0 + i * cell_h, y0 + i * cell_h],
                    color='#cccccc', linewidth=grid_lw * 0.4, transform=t)
            ax.plot([x0 + i * cell_w, x0 + i * cell_w], [y0, y0 + mat_h],
                    color='#cccccc', linewidth=grid_lw * 0.4, transform=t)
        # Block boundaries
        for boundary in block_boundaries:
            ax.plot([x0, x0 + mat_w],
                    [y0 + (n - boundary) * cell_h, y0 + (n - boundary) * cell_h],
                    color='#333333', linewidth=block_lw, linestyle='-', transform=t)
            ax.plot([x0 + boundary * cell_w, x0 + boundary * cell_w],
                    [y0, y0 + mat_h],
                    color='#333333', linewidth=block_lw, linestyle='-', transform=t)
        
        # Shade diagonal blocks
        prev_boundary = 0
        all_boundaries = block_boundaries + [n]
        for boundary in all_boundaries:
            block_size = boundary - prev_boundary
            if block_size > 0:
                rect = plt.Rectangle(
                    (x0 + prev_boundary * cell_w, y0 + (n - boundary) * cell_h),
                    block_size * cell_w, block_size * cell_h,
                    facecolor='#f0f0f0', edgecolor='#999999',
                    linewidth=0.5, alpha=0.3, zorder=-1, transform=t)
                ax.add_patch(rect)
            prev_boundary = boundary
    
    # Outer border
    border_rect = plt.Rectangle((x0, y0), mat_w, mat_h,
                                facecolor='none', edgecolor='#333333',
                                linewidth=border_lw, transform=t)
    ax.add_patch(border_rect)
    
    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    fig.savefig(output_path, dpi=DPI,
                facecolor=colors['background'], edgecolor='none')
    plt.close(fig)




# ============================================================================
# DATASET GENERATION (3-class: yes, s_no, d_no)
# ============================================================================
def generate_property_dataset(
    property_name: str,
    output_dir: str = './graphtrails_dataset',
    n_per_class: int = 50,
    seed: int = 42
) -> str:
    """
    Universal dataset generation function for any property.
    
    Creates 3 directories for each property:
    - yes/: Graphs WITH the property (visual certificate)
    - s_no/: Same layout as YES but WITHOUT the property
    - d_no/: Different layout/visual certificate for NO
    
    Args:
        property_name: Name of property ('cut_vertex', 'connectivity', etc.)
        output_dir: Root directory for dataset
        n_per_class: Number of images per class
        seed: Random seed
        
    Returns:
        Path to the dataset directory
    """
    if property_name not in PROPERTY_CONFIGS:
        raise ValueError(f"Unknown property: {property_name}. Available: {list(PROPERTY_CONFIGS.keys())}")
    
    np.random.seed(seed)
    random.seed(seed)
    
    config = PROPERTY_CONFIGS[property_name]
    
    # Check if s_no/d_no are implemented for this property
    has_s_no = config.get('generate_s_no') is not None
    has_d_no = config.get('generate_d_no') is not None
    
    # Create directory structure
    yes_dir = os.path.join(output_dir, property_name, 'yes')
    s_no_dir = os.path.join(output_dir, property_name, 's_no')
    d_no_dir = os.path.join(output_dir, property_name, 'd_no')
    os.makedirs(yes_dir, exist_ok=True)
    if has_s_no:
        os.makedirs(s_no_dir, exist_ok=True)
    if has_d_no:
        os.makedirs(d_no_dir, exist_ok=True)
    
    # Master labels dictionary
    labels = {}
    
    print("=" * 70)
    print(f"GRAPHTRAILS Dataset Generator - {config['description']} (3-class)")
    print("=" * 70)
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"Property: {property_name}")
    print(f"Images per class: {n_per_class}")
    classes_str = "yes"
    if has_s_no:
        classes_str += ", s_no (same layout)"
    if has_d_no:
        classes_str += ", d_no (different layout)"
    print(f"Classes: {classes_str}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} @ {DPI} DPI")
    print("-" * 70)
    
    # ========================================================================
    # Reasoning texts storage (for MLLM fine-tuning)
    # ========================================================================
    reasoning_texts = {}
    
    # ========================================================================
    # Generate YES class
    # ========================================================================
    print(f"\n[YES CLASS] Generating graphs WITH {property_name}...")
    
    yes_count = 0
    attempts = 0
    
    while yes_count < n_per_class and attempts < n_per_class * 5:
        attempts += 1
        
        # Set unique seed for this image (for reproducible randomization)
        if USE_RANDOMIZATION:
            set_image_seed(yes_count)
            visual_params = sample_visual_params()
            color_scheme = sample_color_scheme()
        else:
            visual_params = None
            color_scheme = None
        
        try:
            # Generate graph
            result = config['generate_yes']()
            
            if config['has_auxiliary_data']:
                G, auxiliary = result
            else:
                G = result
                auxiliary = None
            
            # Layout - may return (pos, extra_info) tuple or just pos
            layout_result = config['layout_yes'](G, auxiliary)
            
            # Handle layout result based on property type
            bold_nodes = None
            bold_edges = None
            
            if property_name in ['bipartite', 'completeness', 'k_colorability']:
                matrix_data = layout_result
                pos = None

                # Validate layout result for matrix-based properties
                if not isinstance(matrix_data, dict) or 'matrix' not in matrix_data:
                    # Provide debugging details to identify why generation failed
                    raise ValueError(f"Invalid matrix layout returned for {property_name}: {matrix_data}. G_nodes={list(G.nodes())}, auxiliary={auxiliary}")
            else:
                # Check if layout returns tuple (pos, extra_info) for bold nodes/edges
                if isinstance(layout_result, tuple) and len(layout_result) == 2:
                    pos, extra_info = layout_result
                    # For cut_vertex, extra_info is the cut vertex to bold (node)
                    if config.get('yes_bold_nodes', False) and extra_info is not None:
                        bold_nodes = {extra_info}
                    # For connectivity, extra_info is the set of edges to bold
                    elif config.get('yes_bold_edges', False) and extra_info is not None:
                        bold_edges = extra_info
                else:
                    pos = layout_result
            
            # Generate filename
            filename = f"{property_name}_yes_{yes_count + 1:03d}.png"
            filepath = os.path.join(yes_dir, filename)
            
            # Use appropriate visualization
            if property_name in ['bipartite', 'non_bipartite', 'completeness', 'k_colorability']:
                # Matrix-based visual proofs
                visualize_matrix(matrix_data, filepath, color_scheme=color_scheme)
            else:
                # Visualize graph with appropriate highlighting
                # Note: hamiltonian uses bold_edges (already set above) for cycle, not red highlight_edges
                visualize_graph(G, pos, filepath, highlight_edges=None, 
                               node_colors=None, bold_nodes=bold_nodes, bold_edges=bold_edges,
                               visual_params=visual_params, color_scheme=color_scheme)
            
            # Record metadata
            labels[filename] = config['get_metadata'](G, auxiliary, label=1)
            
            # Generate reasoning text for MLLM fine-tuning
            if USE_REASONING:
                reasoning_text = generate_reasoning_text(property_name, 'yes', G, auxiliary)
                reasoning_texts[filename] = reasoning_text
                labels[filename]['reasoning'] = reasoning_text
            
            yes_count += 1
            
            if yes_count % 25 == 0 or yes_count == n_per_class:
                print(f"  Progress: {yes_count}/{n_per_class}")
                
        except Exception as e:
            print(f"  Warning: Failed to generate YES image {yes_count + 1}: {e}")
            continue
    
    # ========================================================================
    # Generate NO class images (s_no AND d_no from SAME generated NO graph)
    # ========================================================================
    s_no_count = 0
    d_no_count = 0
    
    if has_s_no or has_d_no:
        print(f"\n[NO CLASS] Generating NO graphs (s_no and d_no from SAME graph)...")
        
        attempts = 0
        target_count = n_per_class
        
        while (s_no_count < target_count or d_no_count < target_count) and attempts < target_count * 5:
            attempts += 1
            
            # Set unique seed for this iteration (use large offset to avoid collision with YES)
            if USE_RANDOMIZATION:
                set_image_seed(10000 + attempts)
                visual_params_sno = sample_visual_params()
                visual_params_dno = sample_visual_params()
                color_scheme_sno = sample_color_scheme()
                color_scheme_dno = sample_color_scheme()
            else:
                visual_params_sno = None
                visual_params_dno = None
                color_scheme_sno = None
                color_scheme_dno = None
            
            try:
                # Generate ONE NO graph using the unified generator
                G, info = config['generate_no']()
                
                # ============================================================
                # Create s_no image from this graph
                # ============================================================
                if has_s_no and s_no_count < target_count:
                    try:
                        # Extract s_no auxiliary from info dict
                        s_no_info_key = config.get('s_no_info_key')
                        if s_no_info_key is None:
                            s_no_auxiliary = info  # Use entire info dict
                        else:
                            s_no_auxiliary = info.get(s_no_info_key)
                        
                        # Get layout
                        if config.get('s_no_has_auxiliary', False):
                            layout_result = config['layout_s_no'](G, s_no_auxiliary)
                        else:
                            layout_result = config['layout_s_no'](G)
                        
                        # Handle layout result - may return (pos, bold_edges) tuple
                        bold_edges = None
                        if isinstance(layout_result, tuple) and len(layout_result) == 2:
                            pos, bold_edges = layout_result
                        else:
                            pos = layout_result
                        
                        # Generate filename
                        filename = f"{property_name}_s_no_{s_no_count + 1:03d}.png"
                        filepath = os.path.join(s_no_dir, filename)
                        
                        # Visualize
                        if property_name in ['bipartite', 'completeness', 'k_colorability']:
                            visualize_matrix(pos, filepath, color_scheme=color_scheme_sno)  # pos is actually matrix_data
                        else:
                            visualize_graph(G, pos, filepath, highlight_edges=None, 
                                           node_colors=None, bold_nodes=None, bold_edges=bold_edges,
                                           visual_params=visual_params_sno, color_scheme=color_scheme_sno)
                        
                        # Record metadata
                        labels[filename] = config['get_metadata'](G, s_no_auxiliary, label=0)
                        
                        # Generate reasoning text for MLLM fine-tuning
                        if USE_REASONING:
                            reasoning_text = generate_reasoning_text(property_name, 's_no', G, s_no_auxiliary)
                            reasoning_texts[filename] = reasoning_text
                            labels[filename]['reasoning'] = reasoning_text
                        
                        s_no_count += 1
                        
                    except Exception as e:
                        print(f"  Warning: Failed to generate s_no image {s_no_count + 1}: {e}")
                
                # ============================================================
                # Create d_no image from the SAME graph
                # ============================================================
                if has_d_no and d_no_count < target_count:
                    try:
                        # Extract d_no auxiliary from info dict
                        d_no_info_key = config.get('d_no_info_key')
                        if d_no_info_key is None:
                            d_no_auxiliary = None  # d_no uses None
                        else:
                            d_no_auxiliary = info.get(d_no_info_key)
                        
                        # Get layout
                        if config.get('d_no_has_auxiliary', False):
                            layout_result = config['layout_d_no'](G, d_no_auxiliary)
                        else:
                            layout_result = config['layout_d_no'](G)
                        
                        # Generate filename
                        filename = f"{property_name}_d_no_{d_no_count + 1:03d}.png"
                        filepath = os.path.join(d_no_dir, filename)
                        
                        # Visualize with different layout
                        if property_name in ['bipartite', 'completeness', 'k_colorability']:
                            visualize_matrix(layout_result, filepath, color_scheme=color_scheme_dno)
                        else:
                            # layout_result may be (pos, extra_info) for d_no generators
                            bold_edges = None
                            bold_nodes = None
                            if isinstance(layout_result, tuple) and len(layout_result) == 2:
                                pos, extra_info = layout_result
                                # Check if this property uses bold_nodes for d_no (e.g., hamiltonian)
                                if config.get('d_no_bold_nodes', False) and extra_info:
                                    bold_nodes = extra_info  # It's a set of nodes to bold
                                else:
                                    bold_edges = extra_info  # It's edges to bold
                            else:
                                pos = layout_result

                            visualize_graph(G, pos, filepath, highlight_edges=None, 
                                           node_colors=None, bold_nodes=bold_nodes, bold_edges=bold_edges,
                                           visual_params=visual_params_dno, color_scheme=color_scheme_dno)
                        
                        # Record metadata
                        labels[filename] = config['get_metadata'](G, d_no_auxiliary, label=0)
                        
                        # Generate reasoning text for MLLM fine-tuning
                        if USE_REASONING:
                            reasoning_text = generate_reasoning_text(property_name, 'd_no', G, d_no_auxiliary)
                            reasoning_texts[filename] = reasoning_text
                            labels[filename]['reasoning'] = reasoning_text
                        
                        d_no_count += 1
                        
                    except Exception as e:
                        print(f"  Warning: Failed to generate d_no image {d_no_count + 1}: {e}")
                
                # Print progress
                if (s_no_count + d_no_count) % 50 == 0:
                    print(f"  Progress: s_no={s_no_count}/{target_count}, d_no={d_no_count}/{target_count}")
                    
            except Exception as e:
                print(f"  Warning: Failed to generate NO graph: {e}")
                continue
        
        print(f"  Final: s_no={s_no_count}/{target_count}, d_no={d_no_count}/{target_count}")
    
    # ========================================================================
    # Save/update master labels file
    # ========================================================================
    labels_path = os.path.join(output_dir, 'property_labels.json')
    
    # Load existing labels if file exists
    existing_labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            existing_labels = json.load(f)
    
    # Merge with new labels
    existing_labels.update(labels)
    
    # Save updated labels
    with open(labels_path, 'w') as f:
        json.dump(existing_labels, f, indent=2)
    
    # ========================================================================
    # Save/update reasoning texts file (for MLLM fine-tuning)
    # ========================================================================
    if USE_REASONING and reasoning_texts:
        reasoning_path = os.path.join(output_dir, 'reasoning_texts.json')
        
        # Load existing reasoning texts if file exists
        existing_reasoning = {}
        if os.path.exists(reasoning_path):
            with open(reasoning_path, 'r') as f:
                existing_reasoning = json.load(f)
        
        # Merge with new reasoning texts
        existing_reasoning.update(reasoning_texts)
        
        # Save updated reasoning texts
        with open(reasoning_path, 'w') as f:
            json.dump(existing_reasoning, f, indent=2)
        
        print(f"\n[MLLM] Saved {len(reasoning_texts)} reasoning texts to reasoning_texts.json")
    
    # ========================================================================
    # Print summary
    # ========================================================================
    total = yes_count + s_no_count + d_no_count
    print("\n" + "=" * 70)
    print(f"✓ {property_name.upper()} DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total images: {total}")
    print(f"  - yes:  {yes_count}")
    if has_s_no:
        print(f"  - s_no: {s_no_count} (same layout)")
    if has_d_no:
        print(f"  - d_no: {d_no_count} (different layout)")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    {property_name}/")
    print(f"      yes/   ({yes_count} images)")
    if has_s_no:
        print(f"      s_no/  ({s_no_count} images)")
    if has_d_no:
        print(f"      d_no/  ({d_no_count} images)")
    print(f"    property_labels.json (updated)")
    if USE_REASONING:
        print(f"    reasoning_texts.json (updated)")
    print("=" * 70)
    
    return output_dir


# ============================================================================
# UTILITY: Dataset statistics
# ============================================================================

def print_dataset_stats(dataset_dir: str):
    """Print statistics about the generated dataset."""
    labels_path = os.path.join(dataset_dir, 'property_labels.json')
    
    if not os.path.exists(labels_path):
        print("No labels file found!")
        return
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    # Group by property
    properties = {}
    for filename, data in labels.items():
        prop = data['property']
        if prop not in properties:
            properties[prop] = {'yes': 0, 's_no': 0, 'd_no': 0, 'no': 0}
        
        if data['label'] == 1:
            properties[prop]['yes'] += 1
        else:
            # Detect s_no or d_no from filename
            if '_s_no_' in filename:
                properties[prop]['s_no'] += 1
            elif '_d_no_' in filename:
                properties[prop]['d_no'] += 1
            else:
                properties[prop]['no'] += 1
    
    print(f"\nDataset Statistics:")
    print(f"Total images: {len(labels)}")
    print(f"\nBy property:")
    for prop, counts in properties.items():
        total = counts['yes'] + counts['s_no'] + counts['d_no'] + counts['no']
        if counts['s_no'] > 0 or counts['d_no'] > 0:
            # Property with s_no/d_no structure
            print(f"  {prop}: {total} ({counts['yes']} YES, {counts['s_no']} s_no, {counts['d_no']} d_no)")
        else:
            # Regular yes/no structure
            print(f"  {prop}: {total} ({counts['yes']} YES, {counts['no']} NO)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRAPHTRAILS Dataset Generator')
    parser.add_argument('--property', type=str, required=True,
                        help='Property to generate dataset for (use "all" for all properties)')
    parser.add_argument('--n_per_class', type=int, default=150,
                        help='Number of images per class (default: 150)')
    parser.add_argument('--output_dir', type=str, default='./graphtrails_dataset',
                        help='Output directory (default: ./graphtrails_dataset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate property argument
    if args.property != 'all' and args.property not in PROPERTY_CONFIGS:
        print(f"Error: Invalid property '{args.property}'")
        print(f"Valid options: all, {', '.join(PROPERTY_CONFIGS.keys())}")
        exit(1)
    
    # Determine which properties to generate
    if args.property == 'all':
        properties_to_generate = list(PROPERTY_CONFIGS.keys())
        print("=" * 70)
        print("GENERATING ALL PROPERTIES (3-class structure)")
        print("=" * 70)
        print(f"Properties: {len(properties_to_generate)}")
        for prop in properties_to_generate:
            config = PROPERTY_CONFIGS[prop]
            has_s_no = config.get('generate_s_no') is not None
            has_d_no = config.get('generate_d_no') is not None
            status = "✓" if (has_s_no and has_d_no) else "○"
            print(f"  {status} {prop}")
        print(f"Images per class: {args.n_per_class}")
        print("=" * 70)
    else:
        properties_to_generate = [args.property]
    
    # Generate dataset for each property using unified generator
    for i, prop in enumerate(properties_to_generate, 1):
        if len(properties_to_generate) > 1:
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(properties_to_generate)}] Generating: {prop.upper()}")
            print(f"{'=' * 70}")
        
        dataset_dir = generate_property_dataset(
            property_name=prop,
            output_dir=args.output_dir,
            n_per_class=args.n_per_class,
            seed=args.seed + i
        )
    
    # Print overall statistics
    print("\n" + "=" * 70)
    print("FINAL DATASET STATISTICS")
    print("=" * 70)
    print_dataset_stats(args.output_dir)
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 70)
    if args.property == 'all':
        print(f"✓ Generated all {len(properties_to_generate)} properties")
        print(f"✓ Total images: {len(properties_to_generate) * args.n_per_class * 2}")
    else:
        print(f"✓ Generated {args.property}")
    print(f"✓ Location: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review images in each property folder")
    print("  2. Train models: jupyter notebook train_models.ipynb")
    print("  3. Evaluate: jupyter notebook evaluate_graphtrails.ipynb")
    print("=" * 70)

