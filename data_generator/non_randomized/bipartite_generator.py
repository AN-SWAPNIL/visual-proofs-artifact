"""
Non-Bipartite Property Generator

Generates graphs with/without odd cycles following GRAPHTRAILS Fig 5-6 conventions:
- YES (Bipartite): Matrix representation with two empty diagonal blocks (Fig 5a)
- NO (Non-Bipartite): Matrix representation with odd cycle sorted first (Fig 6d)

Uses MATRIX REPRESENTATION for visual proof.

Generation strategy (v2):
- All graphs have 20-40 nodes
- NO graph is generated first, then s_no and d_no are derived from it
- YES graph is generated independently
"""

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Set, Optional

# Global node range
MIN_NODES = 20
MAX_NODES = 40


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_odd_cycle(G: nx.Graph) -> Optional[List[int]]:
    """Find an odd cycle in the graph using BFS-based 2-coloring."""
    if nx.is_bipartite(G):
        return None
    
    # Try to find odd cycle via BFS coloring
    for start in G.nodes():
        color = {start: 0}
        queue = [start]
        parent = {start: None}
        
        while queue:
            node = queue.pop(0)
            for neighbor in G.neighbors(node):
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # Found odd cycle - reconstruct it
                    cycle = []
                    # Path from node to common ancestor
                    path1 = [node]
                    curr = node
                    while parent[curr] is not None:
                        curr = parent[curr]
                        path1.append(curr)
                    
                    # Path from neighbor to common ancestor
                    path2 = [neighbor]
                    curr = neighbor
                    while curr not in path1:
                        curr = parent[curr]
                        path2.append(curr)
                    
                    # Combine paths to form cycle
                    common = curr
                    idx = path1.index(common)
                    cycle = path1[:idx+1] + path2[-2::-1]
                    
                    if len(cycle) % 2 == 1 and len(cycle) >= 3:
                        return cycle
    
    # Fallback: use NetworkX cycle finder
    try:
        cycle = nx.find_cycle(G)
        cycle_nodes = [e[0] for e in cycle]
        if len(cycle_nodes) % 2 == 1:
            return cycle_nodes
    except:
        pass
    
    return None


def _generate_random_connected(size: int) -> nx.Graph:
    """Generate a random connected graph."""
    max_attempts = 20
    for _ in range(max_attempts):
        graph_type = random.choice(['erdos_renyi', 'watts_strogatz', 'barabasi_albert'])
        try:
            if graph_type == 'erdos_renyi':
                G = nx.erdos_renyi_graph(size, random.uniform(0.25, 0.45))
            elif graph_type == 'watts_strogatz':
                k = min(4, size - 1)
                G = nx.connected_watts_strogatz_graph(size, k, random.uniform(0.2, 0.4))
            else:
                G = nx.barabasi_albert_graph(size, random.randint(2, 3))
            
            if nx.is_connected(G):
                return G
        except:
            continue
    return nx.cycle_graph(size)


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_bipartite_graph_yes(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Tuple[Set[int], Set[int]]]:
    """
    Generate a BIPARTITE graph (no odd cycles).

    Returns:
        G: Bipartite NetworkX graph
        (set_U, set_V): The two bipartite sets
    """
    # Use global defaults if not specified
    min_n = min_nodes if min_nodes is not None else MIN_NODES
    max_n = max_nodes if max_nodes is not None else MAX_NODES
    
    max_attempts = 100

    for _ in range(max_attempts):
        n = random.randint(min_n, max_n)

        # Split nodes into two sets with more variation
        split_ratio = random.uniform(0.25, 0.75)  # More random split
        n1 = max(1, int(n * split_ratio))
        n2 = n - n1

        # Generate bipartite graph with varied edge probability
        p = random.uniform(0.2, 0.6)  # Wider range for more variation
        try:
            G = nx.bipartite.random_graph(n1, n2, p)
        except Exception:
            continue

        # Try to ensure connectivity by adding edges if necessary without breaking bipartiteness
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            # Color each component separately so we can connect nodes of opposite colors
            comp_colors = {}
            for comp in comps:
                start = next(iter(comp))
                color = {start: 0}
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    for nb in G.neighbors(node):
                        if nb not in color:
                            color[nb] = 1 - color[node]
                            queue.append(nb)
                comp_colors[frozenset(comp)] = color

            for i in range(1, len(comps)):
                comp0 = comps[0]
                compi = comps[i]
                color0 = comp_colors[frozenset(comp0)]
                colori = comp_colors[frozenset(compi)]

                nodes0_color0 = [n for n in comp0 if color0.get(n, 0) == 0]
                nodesi_color1 = [n for n in compi if colori.get(n, 0) == 1]

                if nodes0_color0 and nodesi_color1:
                    a = random.choice(nodes0_color0)
                    b = random.choice(nodesi_color1)
                else:
                    # Fallback: try the opposite pairing
                    nodes0_color1 = [n for n in comp0 if color0.get(n, 0) == 1]
                    nodesi_color0 = [n for n in compi if colori.get(n, 0) == 0]
                    if nodes0_color1 and nodesi_color0:
                        a = random.choice(nodes0_color1)
                        b = random.choice(nodesi_color0)
                    else:
                        # Last resort: pick arbitrary nodes (rare)
                        a = random.choice(list(comp0))
                        b = random.choice(list(compi))

                G.add_edge(a, b)

        # Ensure bipartiteness and extract partitions robustly
        if nx.is_bipartite(G):
            try:
                # Prefer 'bipartite' node attribute if present
                set_U = {n for n, d in G.nodes(data=True) if d.get('bipartite', None) == 0}
                set_V = set(G.nodes()) - set_U

                if len(set_U) == 0 or len(set_V) == 0:
                    set_U, set_V = nx.bipartite.sets(G)
            except Exception:
                # Fallback: compute 2-coloring BFS
                color = {}
                for comp in nx.connected_components(G):
                    start = next(iter(comp))
                    color[start] = 0
                    queue = [start]
                    while queue:
                        node = queue.pop(0)
                        for nb in G.neighbors(node):
                            if nb not in color:
                                color[nb] = 1 - color[node]
                                queue.append(nb)
                set_U = {n for n, c in color.items() if c == 0}
                set_V = set(G.nodes()) - set_U

            # Ensure non-empty partitions and independence within sets (no intra-set edges)
            if len(set_U) > 0 and len(set_V) > 0:
                # Check for intra-set edges which would violate bipartite property for YES graphs
                has_intra = any(v in set_U for u in set_U for v in G.neighbors(u)) or \
                            any(v in set_V for u in set_V for v in G.neighbors(u))
                if has_intra:
                    # Found intra-set edge - try again
                    continue

                return G, (set_U, set_V) 

    # Fallback: complete bipartite graph
    G = nx.complete_bipartite_graph(5, 6)
    set_U = set(range(5))
    set_V = set(range(5, 11))
    return G, (set_U, set_V)  


def generate_bipartite_graph_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a NON-BIPARTITE graph (has odd cycle) - base graph for both s_no and d_no.

    Returns:
        G: Non-bipartite NetworkX graph
        info: Dict containing:
            - 'odd_cycle': List of nodes forming the odd cycle (for d_no layout)
            - 'pseudo_bipartite_sets': Tuple of two sets for s_no layout (partitioning attempt)
    """
    # Use global defaults if not specified
    min_n = min_nodes if min_nodes is not None else MIN_NODES
    max_n = max_nodes if max_nodes is not None else MAX_NODES
    
    max_attempts = 100

    for _ in range(max_attempts):
        n = random.randint(min_n, max_n)

        # Random odd cycle sizes from 3 to min(n//2, 13) - more variety
        max_cycle = min(n // 2, 13)
        possible_cycles = [c for c in range(3, max_cycle + 1, 2)]
        cycle_size = random.choice(possible_cycles) if possible_cycles else 3

        # Create the initial odd cycle
        G = nx.cycle_graph(cycle_size)
        original_cycle = list(range(cycle_size))

        # Add more nodes connected to the cycle
        extra_nodes = n - cycle_size
        for i in range(extra_nodes):
            new_node = cycle_size + i
            G.add_node(new_node)
            # Connect to 2-3 random existing nodes
            num_connections = min(random.randint(2, 3), len(G.nodes()) - 1)
            targets = random.sample(list(G.nodes())[:-1], num_connections)
            for t in targets:
                G.add_edge(new_node, t)

        # Add some random edges for density
        nodes = list(G.nodes())
        for _ in range(n // 3):
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)

        # Verify non-bipartite and connected
        if not nx.is_bipartite(G) and nx.is_connected(G):
            # Create pseudo-sets for s_no layout (approximate 2-partition)
            # Use BFS coloring to get the most balanced "bipartite-like" split
            color = {}
            for comp in nx.connected_components(G):
                start = next(iter(comp))
                color[start] = 0
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    for nb in G.neighbors(node):
                        if nb not in color:
                            color[nb] = 1 - color[node]
                            queue.append(nb)
                        # Note: conflicts are expected (non-bipartite)
            
            set_U = {n for n, c in color.items() if c == 0}
            set_V = set(G.nodes()) - set_U
            
            return G, {'odd_cycle': original_cycle, 'pseudo_bipartite_sets': (set_U, set_V)}

    # Fallback: odd cycle with extra nodes
    n = min_n
    cycle_size = random.choice([5, 7, 9])
    G = nx.cycle_graph(cycle_size)
    for i in range(cycle_size, n):
        G.add_node(i)
        G.add_edge(i, random.randint(0, i-1))
        G.add_edge(i, random.randint(0, i-1))

    set_U = set(range(0, n, 2))
    set_V = set(range(1, n, 2))
    return G, {'odd_cycle': list(range(cycle_size)), 'pseudo_bipartite_sets': (set_U, set_V)}


def generate_bipartite_graph_s_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Tuple[Set[int], Set[int]]]:
    """
    Generate a NON-BIPARTITE graph for SAME layout as YES (matrix with 2 blocks but has intra-edges).
    Uses generate_bipartite_graph_no() and returns pseudo-sets for layout.

    Returns:
        G: Non-bipartite graph
        (set_U, set_V): Pseudo-bipartite sets for matrix layout
    """
    G, info = generate_bipartite_graph_no(min_nodes, max_nodes)
    return G, info['pseudo_bipartite_sets']


def generate_bipartite_graph_d_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a NON-BIPARTITE graph for DIFFERENT layout (odd cycle highlighted).
    Uses generate_bipartite_graph_no() and returns odd_cycle for layout.

    Returns:
        G: Non-bipartite NetworkX graph
        odd_cycle: List of nodes forming the odd cycle
    """
    G, info = generate_bipartite_graph_no(min_nodes, max_nodes)
    return G, info['odd_cycle']



def _create_adjacency_matrix_layout(G: nx.Graph, node_order: List[int], 
                                     highlight_cells: List[Tuple[int, int]] = None,
                                     block_boundaries: List[int] = None) -> Dict:
    """
    COMMON layout function for both YES and NO classes.
    Creates adjacency matrix visualization data.
    
    GRAPHTRAILS Fig 5 & 6: Matrix representation for bipartite/non-bipartite.
    
    Args:
        G: NetworkX graph
        node_order: Order of nodes for rows/columns
        highlight_cells: Optional list of (row, col) cells to highlight (for odd cycle)
        block_boundaries: Optional list of indices marking block divisions
    
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


def layout_bipartite_common(G: nx.Graph, auxiliary, is_non_bipartite: bool) -> Dict:
    """
    COMMON layout function used by both YES (bipartite) and NO (non-bipartite for d_no).

    Args:
        G: NetworkX graph
        auxiliary: For non-bipartite: odd_cycle list; For bipartite: (set_U, set_V)
        is_non_bipartite: True for non-bipartite visualization, False for bipartite visualization

    Returns:
        Dict with matrix data for visualization
    """
    if is_non_bipartite:
        # d_no: Non-bipartite - order by odd cycle first and highlight it
        odd_cycle = auxiliary
        cycle_set = set(odd_cycle)
        other_nodes = sorted([n for n in G.nodes() if n not in cycle_set])

        # Order: cycle nodes first, then remaining nodes (sorted for determinism)
        node_order = list(odd_cycle) + other_nodes

        # Highlight the cycle edges in matrix
        highlight_cells = []
        for i in range(len(odd_cycle)):
            u = odd_cycle[i]
            v = odd_cycle[(i + 1) % len(odd_cycle)]
            idx_u = node_order.index(u)
            idx_v = node_order.index(v)
            highlight_cells.append((idx_u, idx_v))
            highlight_cells.append((idx_v, idx_u))

        # Block boundary after cycle nodes
        block_boundaries = [len(odd_cycle)]
    else:
        # YES / s_no: Bipartite-like ordering - order by two partitions
        set_U, set_V = auxiliary

        # Order: first set U, then set V (sorted for determinism)
        node_order = sorted(set_U) + sorted(set_V)

        # No highlight for bipartite (empty diagonal blocks are the proof)
        highlight_cells = []

        # Block boundary between sets
        block_boundaries = [len(set_U)]

    return _create_adjacency_matrix_layout(G, node_order, highlight_cells, block_boundaries)


def layout_bipartite_yes(G: nx.Graph, odd_cycle: List[int]) -> Dict:
    """
    Layout for non-bipartite (odd-cycle) visualization.
    Uses common layout function with odd-cycle first.
    """
    return layout_bipartite_common(G, odd_cycle, is_non_bipartite=True)


def layout_bipartite_no(G: nx.Graph, bipartite_sets: Tuple[Set[int], Set[int]]) -> Dict:
    """
    GRAPHTRAILS Fig 5a layout for BIPARTITE graphs.
    Uses common layout function.
    """
    return layout_bipartite_common(G, bipartite_sets, is_non_bipartite=False)


# ============================================================================
# METADATA
# ============================================================================

def get_bipartite_metadata(G: nx.Graph, auxiliary, label: int) -> dict:
    """Generate metadata for bipartite property."""
    metadata = {
        'property': 'bipartite',
        'label': label,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'ground_truth': {
            'is_bipartite': label == 1,  # YES = bipartite
            'has_odd_cycle': label == 0
        }
    }

    if label == 0 and auxiliary:  # Non-bipartite (d_no) with odd cycle
        metadata['ground_truth']['odd_cycle_size'] = len(auxiliary)
    elif label == 1 and auxiliary:  # Bipartite with sets
        set_U, set_V = auxiliary
        metadata['ground_truth']['partition_sizes'] = [len(set_U), len(set_V)]

    return metadata
