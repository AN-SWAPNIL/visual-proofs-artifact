"""
k-Colorability Property Generator

Generates k-colorable/non-k-colorable graphs following GRAPHTRAILS Fig 5 conventions:
- YES (4-Colorable): Matrix with ≤4 empty diagonal blocks (like bipartite but with 4 blocks)
- NO (Non-4-Colorable): Matrix with >4 blocks (needs more than 4 colors)

Uses MATRIX REPRESENTATION for visual proof:
- Empty diagonal blocks = independent sets (same color class)
- If ≤4 empty diagonal blocks exist, graph is 4-colorable

Generation Strategy v3 (Enhanced Randomization):
- Graphs have 15-50 nodes (wider range for unpredictability)
- Variable color class sizes and counts
- Per-image unique seeding for randomness
- NO graph generated first, then s_no and d_no derived from SAME NO graph
- s_no: Uses same matrix layout (>4 color blocks)
- d_no: Uses clique certificate visualization
- Node radius fixed at 0.1, assigned after scaling
"""

import numpy as np
import networkx as nx
import random
from typing import List, Tuple, Dict, Set, Optional

# Import randomization config (with fallback)
try:
    from randomization_config import (
        get_random_node_range, sample_edge_probability,
        sample_visual_params, CONFIG
    )
    USE_RANDOMIZATION = True
except ImportError:
    USE_RANDOMIZATION = False

# Global node range for all generators (wider range)
MIN_NODES = 15  # Reduced from 20
MAX_NODES = 50  # Increased from 40


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _greedy_coloring(G: nx.Graph) -> Dict[int, int]:
    """Greedy graph coloring."""
    return nx.greedy_color(G, strategy='largest_first')


def _chromatic_number_upper_bound(G: nx.Graph) -> int:
    """Get upper bound on chromatic number using greedy coloring."""
    coloring = _greedy_coloring(G)
    return max(coloring.values()) + 1 if coloring else 1


def _find_clique(G: nx.Graph, size: int) -> Optional[List[int]]:
    """Find a clique of given size if it exists."""
    cliques = list(nx.find_cliques(G))
    for clique in cliques:
        if len(clique) >= size:
            return clique[:size]
    return None


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_k_colorability_graph_yes(k: int = 4, min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a k-COLORABLE graph (chromatic number ≤ k).
    
    GRAPHTRAILS Fig 5: Matrix representation shows ≤k empty diagonal blocks.
    
    Strategy: Generate graph that is 1, 2, 3, or 4-colorable (all are ≤4-colorable).
    - 1-colorable: Independent set (no edges)
    - 2-colorable: Bipartite graph
    - 3-colorable: 3 independent sets with edges between them
    - 4-colorable: 4 independent sets with edges between them
    
    Args:
        k: Target number of colors (default 4)
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: k-colorable NetworkX graph
        coloring: Dict mapping node -> color (0 to actual_k-1)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 100
    
    for _ in range(max_attempts):
        n = random.randint(min_nodes, max_nodes)
        
        # Randomly choose actual number of colors with bias: prefer 4 > 3 > 2 > 1
        # Weights: 4 (50%), 3 (30%), 2 (15%), 1 (5%)
        actual_k = random.choices([4, 3, 2, 1], weights=[0.5, 0.3, 0.15, 0.05], k=1)[0]
        
        if actual_k == 1:
            # 1-colorable: Independent set (no edges) - rare case
            G = nx.Graph()
            for i in range(n):
                G.add_node(i)
            coloring = {i: 0 for i in range(n)}
            return G, coloring
            
        elif actual_k == 2:
            # Bipartite graph (2-colorable)
            n1 = random.randint(n // 3, 2 * n // 3)
            n2 = n - n1
            G = nx.bipartite.random_graph(n1, n2, random.uniform(0.3, 0.5))
            
            if nx.is_connected(G):
                # Get 2-coloring from bipartite sets
                set_U, set_V = nx.bipartite.sets(G)
                coloring = {node: 0 for node in set_U}
                coloring.update({node: 1 for node in set_V})
                return G, coloring
                
        elif actual_k == 3:
            # 3-colorable graph: Create 3 independent sets with edges between them
            sizes = [n // 3, n // 3, n - 2 * (n // 3)]
            
            G = nx.Graph()
            color_assignment = {}
            node_id = 0
            
            # Create nodes in 3 groups
            groups = []
            for color, size in enumerate(sizes):
                group = list(range(node_id, node_id + size))
                groups.append(group)
                for node in group:
                    G.add_node(node)
                    color_assignment[node] = color
                node_id += size
            
            # Add edges only BETWEEN groups (not within)
            for i in range(actual_k):
                for j in range(i + 1, actual_k):
                    for u in groups[i]:
                        for v in groups[j]:
                            if random.random() < 0.35:
                                G.add_edge(u, v)
            
            if nx.is_connected(G):
                return G, color_assignment
        
        else:  # actual_k == 4
            # 4-colorable: 4 independent sets
            sizes = [n // 4] * 4
            sizes[-1] += n - sum(sizes)
            
            G = nx.Graph()
            color_assignment = {}
            node_id = 0
            
            groups = []
            for color, size in enumerate(sizes):
                group = list(range(node_id, node_id + size))
                groups.append(group)
                for node in group:
                    G.add_node(node)
                    color_assignment[node] = color
                node_id += size
            
            # Add edges between groups
            for i in range(4):
                for j in range(i + 1, 4):
                    for u in groups[i]:
                        for v in groups[j]:
                            if random.random() < 0.3:
                                G.add_edge(u, v)
            
            if nx.is_connected(G):
                return G, color_assignment
    
    # Fallback: simple 4-colorable graph (cycle of length 12)
    G = nx.cycle_graph(12)
    coloring = {i: i % 4 for i in range(12)}
    return G, coloring


def generate_k_colorability_graph_no(k: int = 4, min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a NON-k-COLORABLE graph (requires k+1 or more colors).
    This is the UNIFIED generator - s_no and d_no are derived from the same graph.
    
    GRAPHTRAILS: Contains large clique which proves non-k-colorability.
    
    Strategy: Create k+1 to k+4 independent sets (color classes), then add
    edges BETWEEN sets (not within). The clique nodes each get their own color,
    extra nodes are distributed into these color classes.
    
    This ensures:
    1. Graph needs exactly num_colors colors (chromatic number = num_colors)
    2. Each color class is an independent set (empty diagonal blocks)
    3. Clique nodes prove non-k-colorability
    
    Args:
        k: Target number of colors (default 4)
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-k-colorable NetworkX graph
        info: Dict containing:
            - 'coloring': Dict mapping node -> color for s_no layout
            - 'clique_nodes': List of nodes in clique for d_no layout
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 50
    
    # Randomize number of colors needed: 5, 6, 7, or 8 (for k=4)
    num_colors = random.randint(k + 1, k + 4)  # 5-8 colors
    
    for _ in range(max_attempts):
        n = random.randint(min_nodes, max_nodes)
        
        # Create num_colors independent sets (color classes)
        # First num_colors nodes form the clique (each in its own color class)
        # Remaining nodes are distributed among color classes
        
        G = nx.Graph()
        coloring = {}
        color_classes = [[] for _ in range(num_colors)]
        
        # Clique nodes: each gets its own color (0 to num_colors-1)
        clique_nodes = list(range(num_colors))
        for i, node in enumerate(clique_nodes):
            G.add_node(node)
            coloring[node] = i
            color_classes[i].append(node)
        
        # Add clique edges (all pairs in clique)
        for i in range(num_colors):
            for j in range(i + 1, num_colors):
                G.add_edge(clique_nodes[i], clique_nodes[j])
        
        # Distribute remaining nodes among color classes
        extra_nodes = n - num_colors
        node_id = num_colors
        
        for i in range(extra_nodes):
            # Assign to a random color class
            color = random.randint(0, num_colors - 1)
            G.add_node(node_id)
            coloring[node_id] = color
            color_classes[color].append(node_id)
            node_id += 1
        
        # Add edges BETWEEN different color classes (not within)
        # This maintains the independent set property
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                nodes1 = color_classes[c1]
                nodes2 = color_classes[c2]
                
                # Add random edges between these color classes
                for u in nodes1:
                    for v in nodes2:
                        # Higher probability for clique connections, lower for others
                        if u in clique_nodes and v in clique_nodes:
                            continue  # Already added
                        elif u in clique_nodes or v in clique_nodes:
                            # Connect extra nodes to clique with moderate probability
                            if random.random() < 0.4:
                                G.add_edge(u, v)
                        else:
                            # Connect extra nodes to each other with lower probability
                            if random.random() < 0.25:
                                G.add_edge(u, v)
        
        # Ensure connectivity
        if not nx.is_connected(G):
            # Connect components by adding edges between different color classes
            components = list(nx.connected_components(G))
            for i in range(1, len(components)):
                comp0 = list(components[0])
                compi = list(components[i])
                
                # Find nodes from different color classes to connect
                for u in comp0:
                    for v in compi:
                        if coloring[u] != coloring[v]:  # Different colors - can add edge
                            G.add_edge(u, v)
                            break
                    else:
                        continue
                    break
        
        if nx.is_connected(G):
            return G, {'coloring': coloring, 'clique_nodes': clique_nodes}
    
    # Fallback: simple complete graph with num_colors nodes
    G = nx.complete_graph(num_colors)
    clique_nodes = list(range(num_colors))
    coloring = {i: i for i in range(num_colors)}
    return G, {'coloring': coloring, 'clique_nodes': clique_nodes}


# New generators for SAME-LAYOUT NO (s_no) and DIFFERENT-LAYOUT NO (d_no)
def generate_k_colorability_graph_s_no(k: int = 4, min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a SAME-LAYOUT NON-k-COLORABLE graph.
    Derived from the unified NO generator.
    Uses same matrix layout (>4 color blocks).

    Args:
        k: Target number of colors (default 4)
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G, coloring dict mapping node -> color (color indices 0..r-1, where r>k)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract s_no data
    G, info = generate_k_colorability_graph_no(k, min_nodes, max_nodes)
    return G, info['coloring']


def generate_k_colorability_graph_d_no(k: int = 4, min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a DIFFERENT-LAYOUT NON-k-COLORABLE graph with clique certificate.
    Derived from the unified NO generator.
    Uses clique certificate visualization.

    Args:
        k: Target number of colors (default 4)
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: NetworkX graph containing a clique of size 5-8
        clique_nodes: List of nodes forming the clique (to be highlighted in layout)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract d_no data
    G, info = generate_k_colorability_graph_no(k, min_nodes, max_nodes)
    return G, info['clique_nodes']


def layout_k_colorability_d_no(G: nx.Graph, clique_nodes: List[int]) -> Dict:
    """
    Layout for d_no (non-k-colorable with 5-clique certificate) visualization.
    
    Places 5-clique nodes first in the matrix, then remaining nodes.
    Highlights all clique edges as the certificate of non-4-colorability.
    
    This mirrors bipartite_generator's layout_bipartite_common for odd cycles,
    but shows a 5-clique instead.
    
    Returns:
        Dict with matrix data for visualization
    """
    clique_set = set(clique_nodes)
    other_nodes = sorted([n for n in G.nodes() if n not in clique_set])
    
    # Order: clique nodes first, then remaining nodes (sorted for determinism)
    node_order = list(clique_nodes) + other_nodes
    
    # Highlight all clique edges in matrix (complete graph, so all pairs)
    highlight_cells = []
    for i in range(len(clique_nodes)):
        for j in range(i + 1, len(clique_nodes)):
            u = clique_nodes[i]
            v = clique_nodes[j]
            idx_u = node_order.index(u)
            idx_v = node_order.index(v)
            highlight_cells.append((idx_u, idx_v))
            highlight_cells.append((idx_v, idx_u))
    
    # Block boundary after clique nodes
    block_boundaries = [len(clique_nodes)]
    
    return _create_adjacency_matrix_layout(G, node_order, highlight_cells, block_boundaries)


# ============================================================================
# LAYOUT FUNCTIONS - Matrix Representation
# ============================================================================

def _create_adjacency_matrix_layout(G: nx.Graph, node_order: List[int], 
                                     highlight_cells: List[Tuple[int, int]] = None,
                                     block_boundaries: List[int] = None) -> Dict:
    """
    Creates adjacency matrix visualization data.
    
    GRAPHTRAILS Fig 5: Matrix representation for k-colorability.
    - Empty diagonal blocks = independent sets (same color, no edges within)
    - ≤4 blocks proves 4-colorability
    
    Args:
        G: NetworkX graph
        node_order: Order of nodes for rows/columns
        highlight_cells: Optional list of (row, col) cells to highlight
        block_boundaries: List of indices marking block divisions (color group boundaries)
    
    Returns:
        Dict with matrix data for visualization
    """
    n = len(node_order)
    node_to_idx = {node: i for i, node in enumerate(node_order)}
    
    # Create adjacency matrix
    matrix = np.zeros((n, n), dtype=int)
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            matrix[i, j] = 1
            matrix[j, i] = 1
    
    return {
        'matrix': matrix,
        'node_order': node_order,
        'highlight_cells': highlight_cells or [],
        'block_boundaries': block_boundaries or [],
        'is_matrix': True
    }


def layout_k_colorability_yes(G: nx.Graph, coloring: Dict[int, int]) -> Dict:
    """
    GRAPHTRAILS layout for k-COLORABLE graphs (≤4 colors).
    
    Matrix representation with ≤4 empty diagonal blocks.
    Each block = one color class (independent set, no edges within).
    
    Visual proof: Empty diagonal blocks show valid k-coloring.
    
    Returns:
        Dict with matrix data for visualization
    """
    # Group nodes by color
    color_groups = {}
    for node, color in coloring.items():
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(node)
    
    # Order nodes by color groups (creates block structure)
    node_order = []
    block_boundaries = []
    
    for color_idx in sorted(color_groups.keys()):
        node_order.extend(color_groups[color_idx])
        block_boundaries.append(len(node_order))
    
    # Remove last boundary (it's just the total length)
    if block_boundaries:
        block_boundaries = block_boundaries[:-1]
    
    return _create_adjacency_matrix_layout(G, node_order, [], block_boundaries)


def layout_k_colorability_no(G: nx.Graph, coloring: Dict[int, int]) -> Dict:
    """
    GRAPHTRAILS layout for NON-k-COLORABLE graphs (needs >4 colors).
    
    Matrix representation with >4 blocks (5, 6, 7, or 8 blocks).
    Shows that graph requires more than 4 colors.
    
    The coloring from generate_k_colorability_graph_no is GUARANTEED to create
    proper independent sets (no edges within each color class).
    
    Returns:
        Dict with matrix data for visualization (with optional 'highlight_cells')
    """
    # Group nodes by color
    color_groups = {}
    for node, color in coloring.items():
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(node)
    
    # Order nodes by color groups (creates block structure with empty diagonal blocks)
    node_order = []
    block_boundaries = []
    
    for color_idx in sorted(color_groups.keys()):
        node_order.extend(sorted(color_groups[color_idx]))  # Sort for determinism
        block_boundaries.append(len(node_order))
    
    # Remove last boundary
    if block_boundaries:
        block_boundaries = block_boundaries[:-1]

    # Attempt to find a large clique (size 5+) to highlight as an explicit certificate
    clique = _find_clique(G, size=5)
    highlight_cells = []
    if clique:
        node_to_idx = {node: i for i, node in enumerate(node_order)}
        indices = [node_to_idx[n] for n in clique if n in node_to_idx]
        for i in indices:
            for j in indices:
                if i != j:  # Don't highlight diagonal
                    highlight_cells.append((i, j))

    return _create_adjacency_matrix_layout(G, node_order, highlight_cells, block_boundaries)


# ============================================================================
# METADATA
# ============================================================================

def get_k_colorability_metadata(G: nx.Graph, auxiliary, label: int) -> dict:
    """Generate metadata for k-colorability property."""
    chromatic_bound = _chromatic_number_upper_bound(G)
    
    metadata = {
        'property': 'k_colorability',
        'label': label,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'ground_truth': {
            'chromatic_number_upper_bound': chromatic_bound
        }
    }
    
    if auxiliary:
        # YES and s_no return coloring dict, d_no returns clique_nodes list
        if isinstance(auxiliary, dict):
            metadata['ground_truth']['num_colors_used'] = len(set(auxiliary.values()))
        elif isinstance(auxiliary, list):
            # d_no: auxiliary is the clique nodes list
            metadata['ground_truth']['clique_size'] = len(auxiliary)
            metadata['ground_truth']['is_partial'] = True  # d_no visualization is partial
    
    return metadata
