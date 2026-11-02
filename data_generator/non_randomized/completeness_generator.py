"""
Completeness Property Generator

Generates complete/non-complete graphs following GRAPHTRAILS Fig 7 conventions:
- YES (Complete): All possible edges present - circular layout
- NO (Non-Complete): Missing edge highlighted - circular layout or matrix

GRAPHTRAILS: Matrix representation scales well for spotting missing edges.

Generation Strategy v2:
- All graphs generated with 20-40 nodes (more randomized)
- NO graph generated first, then s_no and d_no derived from SAME NO graph
- s_no: Uses same matrix layout showing missing edge
- d_no: Uses matrix with ALL missing cells shaded
- Node radius fixed at 0.1, assigned after scaling
"""

import numpy as np
import networkx as nx
import random
from typing import List, Tuple, Dict, Set, Optional

# Global node range for all generators
MIN_NODES = 20
MAX_NODES = 40


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_completeness_graph_yes(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, None]:
    """
    Generate a COMPLETE graph (all possible edges present).
    
    GRAPHTRAILS Fig 7c: Matrix shows full grid (except diagonal).
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Complete NetworkX graph
        None: No auxiliary data needed
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    n = random.randint(min_nodes, max_nodes)
    G = nx.complete_graph(n)
    return G, None


def generate_completeness_graph_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a NON-COMPLETE graph (at least one edge missing).
    This is the UNIFIED generator - s_no and d_no are derived from the same graph.
    
    GRAPHTRAILS Fig 7b/7c: Missing edge clearly visible in matrix.
    
    Strategy: Start with complete graph, remove 1-3 edges.
    Can be single component or multiple components.
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-complete NetworkX graph
        info: Dict containing:
            - 'missing_edge': Tuple (u, v) of a missing edge for s_no layout
            - 'all_missing_edges': List of all missing edges for d_no layout
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 50
    
    for _ in range(max_attempts):
        n = random.randint(min_nodes, max_nodes)
        G = nx.complete_graph(n)
        
        # Calculate total possible edges
        total_edges = n * (n - 1) // 2
        
        # Remove random percentage of edges (5% to 30%) for high variation
        # This creates graphs with vastly different numbers of missing edges
        removal_percentage = random.uniform(0.05, 0.30)
        num_to_remove = max(1, int(total_edges * removal_percentage))
        
        # Add some noise to make it even more random
        num_to_remove += random.randint(-3, 3)
        num_to_remove = max(1, min(num_to_remove, total_edges - 1))  # At least 1, at most total-1
        
        edges = list(G.edges())
        random.shuffle(edges)  # Randomize edge order
        
        removed_edges = []
        for i in range(num_to_remove):
            if i < len(edges):
                edge = edges[i]
                G.remove_edge(*edge)
                removed_edges.append(edge)
        
        if removed_edges:
            # Return info for both s_no and d_no layouts
            return G, {
                'missing_edge': removed_edges[0],
                'all_missing_edges': removed_edges
            }
    
    # Fallback: K_25 with multiple edges removed
    G = nx.complete_graph(25)
    n_remove = random.randint(10, 50)
    edges = list(G.edges())
    random.shuffle(edges)
    removed = edges[:n_remove]
    for edge in removed:
        G.remove_edge(*edge)
    return G, {
        'missing_edge': removed[0],
        'all_missing_edges': removed
    }


def generate_completeness_graph_s_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Tuple[int, int]]:
    """
    Generate a NON-COMPLETE graph (same layout as YES).
    Derived from the unified NO generator.
    Uses same matrix layout showing missing edge.
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-complete NetworkX graph
        missing_edge: Tuple (u, v) of a missing edge
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract s_no data
    G, info = generate_completeness_graph_no(min_nodes, max_nodes)
    return G, info['missing_edge']


def generate_completeness_graph_d_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """
    Generate a NON-COMPLETE graph (different layout visualization).
    Derived from the unified NO generator.
    Uses matrix with ALL missing cells shaded.
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-complete NetworkX graph
        all_missing_edges: List of all missing edges for shading
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract d_no data
    G, info = generate_completeness_graph_no(min_nodes, max_nodes)
    return G, info['all_missing_edges']


# ============================================================================
# LAYOUT FUNCTIONS (MATRIX REPRESENTATION)
# ============================================================================

def _create_adjacency_matrix_layout(G: nx.Graph, 
                                     highlight_missing: Optional[List[Tuple[int, int]]] = None,
                                     shade_missing: bool = False) -> Dict:
    """
    Create adjacency matrix representation for completeness visualization.

    GRAPHTRAILS Fig 7c: Matrix representation scales well for spotting missing edges.
    Grid lines between every row/column make cells clearly visible.

    Args:
        G: The graph
        highlight_missing: List of missing edges to highlight (for non-complete graphs)
        shade_missing: If True, shade all empty (zero) off-diagonal cells to emphasize non-completeness

    Returns:
        Dictionary with matrix visualization data
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Create adjacency matrix
    matrix = np.zeros((n, n), dtype=int)
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        matrix[i, j] = 1
        matrix[j, i] = 1

    # Identify missing edges for non-complete graphs (specific ones to highlight)
    missing_cells = []
    if highlight_missing:
        for u, v in highlight_missing:
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                missing_cells.append((i, j))
                missing_cells.append((j, i))

    # Shade all empty off-diagonal cells when requested (d_no behavior)
    shade_missing_cells = []
    if shade_missing:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # Keep diagonal empty/ignored
                if matrix[i, j] == 0:
                    shade_missing_cells.append((i, j))

    # Create block boundaries for every row/column (grid lines)
    block_boundaries = list(range(1, n))

    return {
        'matrix': matrix,
        'node_order': nodes,
        'highlight_cells': [],
        'block_boundaries': block_boundaries,  # Grid lines between every row/column
        'missing_cells': missing_cells,
        'shade_missing_cells': shade_missing_cells,
        'is_matrix': True
    }


def layout_completeness_common(G: nx.Graph, 
                                auxiliary, 
                                is_non_complete: bool) -> Dict:
    """
    Common layout function for both complete and non-complete graphs.
    
    Uses matrix representation per GRAPHTRAILS recommendations.
    
    Args:
        G: The graph
        auxiliary: Missing edge tuple for non-complete, None for complete
        is_non_complete: True for non-complete (NO), False for complete (YES)
    
    Returns:
        Dictionary with matrix layout data
    """
    if is_non_complete and auxiliary:
        # Highlight the missing edge(s)
        return _create_adjacency_matrix_layout(G, highlight_missing=[auxiliary])
    else:
        # Complete graph - no missing edges
        return _create_adjacency_matrix_layout(G, highlight_missing=None)


def layout_completeness_yes(G: nx.Graph, auxiliary=None) -> Dict:
    """
    GRAPHTRAILS matrix layout for COMPLETE graphs.
    
    Visual proof: Full matrix (except diagonal) - all possible edges present.
    """
    return layout_completeness_common(G, auxiliary, is_non_complete=False)


def layout_completeness_no(G: nx.Graph, missing_edge: Tuple[int, int]) -> Dict:
    """
    GRAPHTRAILS Fig 7c matrix layout for NON-COMPLETE graphs.

    Visual proof: Missing cells in matrix clearly show missing edges.
    """
    return layout_completeness_common(G, missing_edge, is_non_complete=True)


def layout_completeness_d_no(G: nx.Graph, missing_edges: List[Tuple[int, int]]) -> Dict:
    """
    DIFFERENT-layout NO for completeness: Shade ALL empty (off-diagonal) cells in the matrix.

    This makes the absence of many edges visually prominent (black/grey cells),
    while highlighting the first few missing edges as examples.
    """
    # Highlight the first missing edge as example, shade all empty cells
    first_edge = missing_edges[0] if missing_edges else None
    return _create_adjacency_matrix_layout(G, highlight_missing=[first_edge] if first_edge else [], shade_missing=True)


# ============================================================================
# METADATA
# ============================================================================

def get_completeness_metadata(G: nx.Graph, auxiliary, label: int) -> dict:
    """Generate metadata for completeness property."""
    n = G.number_of_nodes()
    max_edges = n * (n - 1) // 2
    actual_edges = G.number_of_edges()
    
    metadata = {
        'property': 'completeness',
        'label': label,
        'num_nodes': n,
        'num_edges': actual_edges,
        'ground_truth': {
            'is_complete': label == 1,
            'max_possible_edges': max_edges,
            'missing_edges': max_edges - actual_edges
        }
    }
    
    if label == 0 and auxiliary:  # Non-complete with missing edge
        metadata['ground_truth']['example_missing_edge'] = list(auxiliary)
    
    return metadata
