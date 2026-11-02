"""
Hamiltonian Property Generator

Generates graphs with/without Hamiltonian cycles following GRAPHTRAILS conventions:
- YES (Hamiltonian): All nodes on outer circle (the Hamiltonian cycle)
- NO (Non-Hamiltonian): Longest cycle on outer circle, remaining nodes inside

Both use SAME layout: longest/max cycle on outer boundary with RED highlighting.
Visual difference: YES has all nodes on circle, NO has some nodes inside.

Generation Strategy v3 (Enhanced Randomization):
- Graphs have 12-45 nodes (wider range for unpredictability)
- Multiple non-Hamiltonian construction strategies
- Per-image unique seeding for randomness
- NO graph generated first, then s_no and d_no derived from SAME NO graph
- s_no: Uses same layout (longest cycle on outer circle)
- d_no: Uses cut-set visualization showing k(G-S) > |S|
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
MIN_NODES = 12  # Reduced from 20
MAX_NODES = 45  # Increased from 40


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_longest_cycle(G: nx.Graph, max_time: int = 2000) -> Optional[List[int]]:
    """
    Find the longest cycle in the graph using heuristic search.
    Returns list of nodes in cycle order.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    if n < 3:
        return None
    
    best_cycle = None
    best_length = 0
    attempts = [0]
    
    def find_cycle_from(start: int, target_length: int) -> Optional[List[int]]:
        """Try to find a cycle of at least target_length starting from start."""
        attempts[0] += 1
        if attempts[0] > max_time:
            return None
        
        def backtrack(path: List[int], visited: Set[int]) -> Optional[List[int]]:
            attempts[0] += 1
            if attempts[0] > max_time:
                return None
            
            current = path[-1]
            
            # Check if we can complete a cycle back to start
            if len(path) >= 3 and G.has_edge(current, start):
                if len(path) >= target_length:
                    return path[:]
            
            # Continue exploring
            neighbors = list(G.neighbors(current))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    path.append(neighbor)
                    visited.add(neighbor)
                    
                    result = backtrack(path, visited)
                    if result and len(result) >= target_length:
                        return result
                    
                    path.pop()
                    visited.remove(neighbor)
            
            return None
        
        path = [start]
        visited = {start}
        return backtrack(path, visited)
    
    # Try to find progressively longer cycles
    random.shuffle(nodes)
    
    for start in nodes[:min(10, len(nodes))]:
        # First find any cycle from this node
        for target in range(n, 2, -1):  # Start from longest possible
            if attempts[0] > max_time:
                break
            cycle = find_cycle_from(start, target)
            if cycle and len(cycle) > best_length:
                best_cycle = cycle
                best_length = len(cycle)
                if best_length == n:  # Found Hamiltonian cycle
                    return best_cycle
                break  # Found best from this start, try another
        
        attempts[0] = 0  # Reset for next start
    
    return best_cycle


def _has_hamiltonian_cycle_heuristic(G: nx.Graph, max_time: int = 1000) -> Tuple[bool, Optional[List[int]]]:
    """
    Heuristic check for Hamiltonian cycle using backtracking with pruning.
    Returns (has_cycle, cycle_nodes) if found within time limit.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    if n < 3:
        return False, None
    
    # Check necessary conditions
    if G.number_of_edges() < n:
        return False, None
    
    # Check minimum degree (Ore's theorem heuristic)
    min_degree = min(dict(G.degree()).values())
    if min_degree < 2:
        return False, None
    
    # Backtracking with time limit
    attempts = [0]
    
    def backtrack(path: List[int], visited: Set[int]) -> Optional[List[int]]:
        attempts[0] += 1
        if attempts[0] > max_time:
            return None
        
        if len(path) == n:
            # Check if we can return to start
            if G.has_edge(path[-1], path[0]):
                return path
            return None
        
        current = path[-1]
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)  # Randomize for variety
        
        for neighbor in neighbors:
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)
                
                result = backtrack(path, visited)
                if result:
                    return result
                
                path.pop()
                visited.remove(neighbor)
        
        return None
    
    # Try from random starting nodes
    random.shuffle(nodes)
    for start in nodes[:5]:  # Try up to 5 starting nodes
        path = [start]
        visited = {start}
        result = backtrack(path, visited)
        if result:
            return True, result
        attempts[0] = 0  # Reset for next start
    
    return False, None


def _find_cut_set(G: nx.Graph) -> Tuple[Set[int], List[Set[int]]]:
    """
    Find a cut-set S where k(G-S) > |S| (violates Hamiltonian theorem).
    Uses heuristic search for small cut-sets that fragment the graph.
    
    CRITICAL: Must ensure |S| < k(G-S) for valid counter-example.
    
    Returns:
        (cut_set, components): The cut-set S and resulting components when S is removed
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    best_cut = None
    best_score = 0  # Score = (k(G-S) - |S|) * |S| - prefer larger cut-sets with good ratio
    
    # Try cut-sets of size 2, 3, 4 FIRST (prefer larger cut-sets), then 1 as fallback
    for s_size in [3, 4, 2, 1]:
        # Try random subsets - more attempts for preferred sizes
        attempts = min(200, n * 15) if s_size >= 2 else min(50, n * 5)
        for _ in range(attempts):
            if len(nodes) < s_size:
                continue
            cut_set = set(random.sample(nodes, s_size))
            
            # Count components when cut_set removed
            G_temp = G.copy()
            for s_node in cut_set:
                if s_node in G_temp:
                    G_temp.remove_node(s_node)
            
            if G_temp.number_of_nodes() > 0:
                components = list(nx.connected_components(G_temp))
                num_components = len(components)
                
                # MUST satisfy: k(G-S) > |S|
                if num_components > len(cut_set):
                    # Score favors: more components, larger cut-set size
                    # Formula: (components - cut_size) + cut_size * 0.5
                    # This prefers larger cut-sets when ratio is similar
                    score = (num_components - len(cut_set)) + len(cut_set) * 0.8
                    if score > best_score:
                        best_score = score
                        best_cut = (cut_set, [set(c) for c in components])
        
        # If we found a good cut-set with size >= 2, prefer it
        if best_cut and len(best_cut[0]) >= 2:
            return best_cut
    
    if best_cut:
        return best_cut
    
    # Fallback: Return articulation points as cut-set
    try:
        art_points = list(nx.articulation_points(G))
        if art_points:
            # Try single articulation point
            cut_set = {art_points[0]}
            G_temp = G.copy()
            G_temp.remove_node(art_points[0])
            components = [set(c) for c in nx.connected_components(G_temp)]
            if len(components) > 1:  # Satisfies k(G-S) > |S| (2 > 1)
                return cut_set, components
    except:
        pass
    
    # Final fallback: pick high-degree nodes
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    # Try combinations until we find valid cut-set
    for s_size in [1, 2, 3]:
        cut_set = set(sorted_nodes[:s_size])
        G_temp = G.copy()
        for node in cut_set:
            if node in G_temp:
                G_temp.remove_node(node)
        if G_temp.number_of_nodes() > 0:
            components = [set(c) for c in nx.connected_components(G_temp)]
            if len(components) > len(cut_set):
                return cut_set, components
    
    # Worst case: return something
    cut_set = {sorted_nodes[0]}
    G_temp = G.copy()
    G_temp.remove_node(sorted_nodes[0])
    components = [set(c) for c in nx.connected_components(G_temp)] if G_temp.number_of_nodes() > 0 else []
    return cut_set, components


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_hamiltonian_graph_yes(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a graph WITH Hamiltonian cycle.
    
    Strategy: Start with cycle, add random chords.
    All nodes will be on the Hamiltonian cycle.
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Graph with Hamiltonian cycle
        cycle: List of nodes in Hamiltonian cycle order (all nodes)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 50
    
    for _ in range(max_attempts):
        n = random.randint(min_nodes, max_nodes)
        
        # Strategy: Start with a cycle (guaranteed Hamiltonian)
        G = nx.cycle_graph(n)
        cycle = list(range(n))  # The original cycle - contains ALL nodes
        
        # Add random chords (edges between non-adjacent cycle nodes)
        num_chords = random.randint(n // 4, n // 2)
        nodes = list(G.nodes())
        
        for _ in range(num_chords):
            # Pick two nodes that are not adjacent in the cycle
            u = random.choice(nodes)
            # Find nodes at distance > 1 in cycle
            candidates = [v for v in nodes if v != u 
                         and abs(v - u) > 1 
                         and abs(v - u) < n - 1
                         and not G.has_edge(u, v)]
            
            if candidates:
                v = random.choice(candidates)
                G.add_edge(u, v)
        
        # Verify still has Hamiltonian cycle (it always will since we started with cycle)
        if nx.is_connected(G):
            return G, cycle
    
    # Fallback: simple cycle with few chords
    n = random.randint(8, 12)
    G = nx.cycle_graph(n)
    G.add_edge(0, n // 2)  # One chord
    return G, list(range(n))


def generate_hamiltonian_graph_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a graph WITHOUT Hamiltonian cycle.
    This is the UNIFIED generator - s_no and d_no are derived from the same graph.
    
    Strategy: Create graphs that provably have no Hamiltonian cycle,
    then find the longest cycle for visualization.
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-Hamiltonian graph
        info: Dict containing:
            - 'longest_cycle': List of nodes in the longest cycle for s_no layout
            - 'cut_set_data': (cut_set, components) for d_no layout
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 100
    
    for _ in range(max_attempts):
        n = random.randint(min_nodes, max_nodes)
        
        # More diverse strategies - favor strategies that create good cut-sets
        # 'star_of_cliques' and 'multi_bridge' create graphs with natural multi-node cut-sets
        strategy = random.choices(
            ['petersen_like', 'bipartite_unbalanced', 'bridge', 
             'star_of_cliques', 'multi_bridge', 'sparse_regular', 'tree_plus'],
            weights=[0.1, 0.15, 0.1, 0.25, 0.2, 0.1, 0.1],
            k=1
        )[0]
        
        try:
            if strategy == 'star_of_cliques':
                # Create a "star" of small cliques/cycles connected through a central cut-set
                # This naturally creates graphs where removing 2-3 nodes disconnects into many parts
                num_branches = random.randint(4, 6)  # More branches than cut-set size
                cut_size = random.randint(2, 3)  # Central cut-set
                branch_size = max(3, (n - cut_size) // num_branches)
                
                G = nx.Graph()
                # Add central cut-set nodes
                cut_nodes = list(range(cut_size))
                G.add_nodes_from(cut_nodes)
                # Connect cut-set nodes to each other (make it a clique or path)
                if random.random() < 0.5:
                    for i in range(cut_size):
                        for j in range(i + 1, cut_size):
                            G.add_edge(i, j)
                else:
                    for i in range(cut_size - 1):
                        G.add_edge(i, i + 1)
                
                node_counter = cut_size
                for b in range(num_branches):
                    # Each branch is a small clique or cycle
                    actual_branch_size = min(branch_size + random.randint(-1, 2), n - node_counter)
                    if actual_branch_size < 2:
                        continue
                    
                    branch_nodes = list(range(node_counter, node_counter + actual_branch_size))
                    G.add_nodes_from(branch_nodes)
                    
                    # Connect branch internally (cycle or clique)
                    if random.random() < 0.6 or actual_branch_size <= 4:
                        # Cycle
                        for i in range(len(branch_nodes)):
                            G.add_edge(branch_nodes[i], branch_nodes[(i + 1) % len(branch_nodes)])
                    else:
                        # Near-complete
                        for i in range(len(branch_nodes)):
                            for j in range(i + 1, len(branch_nodes)):
                                if random.random() < 0.7:
                                    G.add_edge(branch_nodes[i], branch_nodes[j])
                    
                    # Connect branch to ONE cut-set node (this is key for the cut-set property)
                    cut_connector = cut_nodes[b % cut_size]
                    branch_connector = random.choice(branch_nodes)
                    G.add_edge(cut_connector, branch_connector)
                    # Maybe add one more edge to cut-set
                    if random.random() < 0.3 and len(branch_nodes) > 1:
                        other_branch = random.choice([bn for bn in branch_nodes if bn != branch_connector])
                        other_cut = random.choice(cut_nodes)
                        G.add_edge(other_cut, other_branch)
                    
                    node_counter += actual_branch_size
                
            elif strategy == 'petersen_like':
                # Petersen graph is non-Hamiltonian but has 9-cycles
                if n >= 10:
                    G = nx.petersen_graph()
                    # Add random nodes
                    for i in range(10, n):
                        G.add_node(i)
                        targets = random.sample(list(range(min(i, 10))), random.randint(2, 3))
                        for t in targets:
                            G.add_edge(i, t)
                else:
                    continue
                    
            elif strategy == 'bipartite_unbalanced':
                # Bipartite graph with unbalanced parts can't be Hamiltonian
                n1 = random.randint(n // 4, n // 3)
                n2 = n - n1
                if abs(n1 - n2) >= 3:  # More unbalanced
                    G = nx.complete_bipartite_graph(n1, n2)
                    # Remove random edges
                    edges = list(G.edges())
                    remove_count = random.randint(len(edges) // 4, len(edges) // 2)
                    for edge in random.sample(edges, min(remove_count, len(edges))):
                        G.remove_edge(*edge)
                        if not nx.is_connected(G):
                            G.add_edge(*edge)
                else:
                    continue
                    
            elif strategy == 'bridge':
                # Graph with a bridge often non-Hamiltonian
                size1 = random.randint(n // 3, 2 * n // 3)
                size2 = n - size1
                
                # Random graph types for each side
                g_type = random.choice(['cycle', 'wheel', 'complete'])
                if g_type == 'cycle':
                    G1 = nx.cycle_graph(size1)
                elif g_type == 'wheel' and size1 >= 4:
                    G1 = nx.wheel_graph(size1)
                else:
                    G1 = nx.complete_graph(size1)
                
                G2 = nx.cycle_graph(size2)
                G2 = nx.relabel_nodes(G2, {i: i + size1 for i in G2.nodes()})
                
                G = nx.compose(G1, G2)
                
                # Connect with 1-2 bridges
                num_bridges = random.randint(1, 2)
                for _ in range(num_bridges):
                    u = random.randint(0, size1 - 1)
                    v = random.randint(size1, n - 1)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                
                # Add some random edges
                for _ in range(random.randint(n // 6, n // 3)):
                    u = random.randint(0, size1 - 1)
                    v = random.randint(size1, n - 1)
                    if not G.has_edge(u, v) and random.random() < 0.4:
                        G.add_edge(u, v)
                        
            elif strategy == 'multi_bridge':
                # Multiple components connected through a small central hub
                # This creates a natural cut-set (the hub nodes)
                num_parts = random.randint(4, 6)  # More parts than hub size
                hub_size = random.randint(2, 3)  # Hub becomes the cut-set
                
                # Calculate sizes for non-hub parts
                remaining = n - hub_size
                part_sizes = []
                for i in range(num_parts - 1):
                    size = random.randint(max(3, remaining // (num_parts - i + 1)), 
                                         max(4, remaining // (num_parts - i)))
                    part_sizes.append(size)
                    remaining -= size
                    if remaining < 3:
                        break
                if remaining >= 3:
                    part_sizes.append(remaining)
                
                G = nx.Graph()
                
                # Create hub (becomes the cut-set)
                hub_nodes = list(range(hub_size))
                G.add_nodes_from(hub_nodes)
                # Connect hub nodes
                for i in range(hub_size - 1):
                    G.add_edge(hub_nodes[i], hub_nodes[i + 1])
                
                offset = hub_size
                for idx, size in enumerate(part_sizes):
                    # Each part is a cycle with extra edges
                    part_nodes = list(range(offset, offset + size))
                    G.add_nodes_from(part_nodes)
                    
                    # Create cycle
                    for i in range(size):
                        G.add_edge(part_nodes[i], part_nodes[(i + 1) % size])
                    
                    # Add some internal edges
                    if size > 4:
                        for _ in range(random.randint(1, size // 3)):
                            u, v = random.sample(part_nodes, 2)
                            G.add_edge(u, v)
                    
                    # Connect to hub (only ONE hub node per part - crucial for cut-set!)
                    hub_connector = hub_nodes[idx % hub_size]
                    part_connector = random.choice(part_nodes)
                    G.add_edge(hub_connector, part_connector)
                    
                    offset += size
                    
            elif strategy == 'sparse_regular':
                # Sparse regular graph - less likely to be Hamiltonian
                d = random.choice([2, 3])
                if n * d % 2 == 0:  # Must be even for regular graph
                    G = nx.random_regular_graph(d, n)
                    # Remove random edges
                    edges = list(G.edges())
                    for _ in range(len(edges) // 5):
                        edge = random.choice(list(G.edges()))
                        G.remove_edge(*edge)
                        if not nx.is_connected(G):
                            G.add_edge(*edge)
                else:
                    continue
                    
            else:  # tree_plus
                # Tree plus few edges - never Hamiltonian if extra edges < n-1
                G = nx.random_tree(n)
                # Add random edges but keep sparse
                num_extra = random.randint(1, n // 4)
                nodes = list(G.nodes())
                for _ in range(num_extra):
                    u, v = random.sample(nodes, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
            
            # Verify non-Hamiltonian (heuristic) and connected
            if nx.is_connected(G):
                has_ham, ham_cycle = _has_hamiltonian_cycle_heuristic(G, max_time=500)
                if not has_ham:
                    # Find the longest cycle for visualization
                    longest_cycle = _find_longest_cycle(G, max_time=1500)
                    if longest_cycle and len(longest_cycle) >= 3:
                        # Generate cut-set data for d_no - MUST satisfy k(G-S) > |S|
                        cut_set_data = _find_cut_set(G)
                        cut_set, components = cut_set_data
                        # Verify the constraint
                        if len(components) > len(cut_set):
                            return G, {'longest_cycle': longest_cycle, 'cut_set_data': cut_set_data}
                    
        except Exception:
            continue
    
    # Fallback: Petersen graph (known non-Hamiltonian, has 9-cycles)
    G = nx.petersen_graph()
    longest_cycle = _find_longest_cycle(G, max_time=2000)
    if not longest_cycle:
        # Petersen graph has 5-cycles easily
        longest_cycle = [0, 1, 2, 3, 4]  # Outer pentagon
    cut_set_data = _find_cut_set(G)
    return G, {'longest_cycle': longest_cycle, 'cut_set_data': cut_set_data}


# ============================================================================
# LAYOUT FUNCTIONS (UNIFIED FOR BOTH YES AND NO)
# ============================================================================

def _layout_with_outer_cycle(G: nx.Graph, cycle: List[int]) -> Tuple[Dict[int, Tuple[float, float]], Set[Tuple[int, int]]]:
    """
    Unified layout: Place cycle nodes on outer circle, other nodes inside.
    
    For YES: All nodes are in cycle, so all on outer circle.
    For NO: Cycle nodes on outer circle, remaining nodes placed inside.
    
    Returns:
        pos: Node positions
        cycle_edges: Set of cycle edges (for bold edge highlighting)
    """
    pos = {}
    n_cycle = len(cycle)
    cycle_set = set(cycle)
    
    # Randomize starting angle for variety
    angle_offset = random.uniform(0, 2 * np.pi)
    
    # Place cycle nodes on outer circle (radius 2.5)
    for i, node in enumerate(cycle):
        angle = angle_offset + 2 * np.pi * i / n_cycle
        x = 2.5 * np.cos(angle)
        y = 2.5 * np.sin(angle)
        pos[node] = (x, y)
    
    # Place non-cycle nodes inside the circle
    other_nodes = [n for n in G.nodes() if n not in cycle_set]
    
    if other_nodes:
        # Use spring layout for interior nodes, constrained to inner area
        if len(other_nodes) > 1:
            # Create subgraph of interior nodes
            interior_subgraph = G.subgraph(other_nodes).copy()
            
            if interior_subgraph.number_of_edges() > 0:
                inner_pos = nx.spring_layout(interior_subgraph, k=0.8, iterations=50, scale=1.2)
            else:
                # No edges among interior nodes - place randomly
                inner_pos = {}
                for node in other_nodes:
                    angle = random.uniform(0, 2 * np.pi)
                    r = random.uniform(0.3, 1.2)
                    inner_pos[node] = (r * np.cos(angle), r * np.sin(angle))
            
            pos.update(inner_pos)
        else:
            # Single interior node - place at center
            for node in other_nodes:
                pos[node] = (0, 0)
    
    # Generate cycle edges for bold highlighting
    cycle_edges = set()
    for i in range(n_cycle):
        u = cycle[i]
        v = cycle[(i + 1) % n_cycle]
        cycle_edges.add((u, v))
    
    return pos, cycle_edges


def layout_hamiltonian_yes(G: nx.Graph, cycle: List[int]) -> Tuple[Dict[int, Tuple[float, float]], Set[Tuple[int, int]]]:
    """
    Layout for HAMILTONIAN graphs.
    
    All nodes are on the outer circle (since cycle includes all nodes).
    Visual proof: Complete cycle on outer boundary with bold edges.
    
    Returns:
        pos: Node positions
        cycle_edges: Set of Hamiltonian cycle edges (for bold edge highlighting)
    """
    return _layout_with_outer_cycle(G, cycle)


def layout_hamiltonian_no(G: nx.Graph, longest_cycle: List[int] = None) -> Tuple[Dict[int, Tuple[float, float]], Set[Tuple[int, int]]]:
    """
    Layout for NON-HAMILTONIAN graphs.
    
    Longest cycle on outer circle, remaining nodes inside.
    Visual proof: Some nodes are INSIDE the circle (not part of longest cycle).
    
    Returns:
        pos: Node positions
        cycle_edges: Set of longest cycle edges (for bold edge highlighting)
    """
    if longest_cycle is None or len(longest_cycle) < 3:
        # Fallback: try to find a cycle
        longest_cycle = _find_longest_cycle(G, max_time=1000)
        if not longest_cycle:
            # Last resort: use spring layout
            return nx.spring_layout(G, k=1.5, iterations=100, scale=2.5, seed=42), set()
    
    return _layout_with_outer_cycle(G, longest_cycle)


# ============================================================================
# SAME-LAYOUT NO (s_no) AND DIFFERENT-LAYOUT NO (d_no)
# ============================================================================

def generate_hamiltonian_graph_s_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a graph WITHOUT Hamiltonian cycle (same layout as YES).
    Derived from the unified NO generator.
    s_no uses the same visualization as original NO - longest cycle on outer circle.
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-Hamiltonian graph
        longest_cycle: List of nodes in the longest cycle (NOT all nodes)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract s_no data
    G, info = generate_hamiltonian_graph_no(min_nodes, max_nodes)
    return G, info['longest_cycle']


def generate_hamiltonian_graph_d_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Tuple[Set[int], List[Set[int]]]]:
    """
    Generate a graph WITHOUT Hamiltonian cycle with cut-set visualization.
    Derived from the unified NO generator.
    
    Uses the theorem: If Hamiltonian, then k(G-S) ≤ |S| for all S.
    Counter-example: Find S where k(G-S) > |S|.
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-Hamiltonian graph
        (cut_set_S, components): Cut-set S and components when S removed
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract d_no data
    G, info = generate_hamiltonian_graph_no(min_nodes, max_nodes)
    return G, info['cut_set_data']


def layout_hamiltonian_s_no(G: nx.Graph, longest_cycle: List[int]) -> Tuple[Dict[int, Tuple[float, float]], Set[Tuple[int, int]]]:
    """
    Layout for s_no (same as original NO layout).
    Longest cycle on outer circle, remaining nodes inside.
    
    Returns:
        pos: Node positions
        cycle_edges: Set of longest cycle edges (for bold edge highlighting)
    """
    return layout_hamiltonian_no(G, longest_cycle)


def layout_hamiltonian_d_no(G: nx.Graph, cut_set_data: Tuple[Set[int], List[Set[int]]]) -> Tuple[Dict[int, Tuple[float, float]], Set[int]]:
    """
    Layout for d_no showing cut-set S proving non-Hamiltonicity.
    
    Like cut-vertex d_no, we just position the nodes WITHOUT drawing extra lines.
    - Cut-set S at CENTER (bold black nodes) - clustered closely together
    - Each component positioned radially using spring layout on subgraph
    - Components are compact clusters around their center
    - Clear visual: |S| nodes at center, >|S| separated component clusters
    
    Returns:
        pos: Node positions
        bold_nodes: Set S (for bold black visualization)
    """
    if not cut_set_data:
        # Fallback
        pos = nx.spring_layout(G, k=1.5, iterations=100, scale=2.5, seed=42)
        return pos, set()
    
    cut_set, components = cut_set_data
    pos = {}
    
    # Place cut-set S at center in a TIGHT cluster
    s_list = sorted(cut_set)
    s_size = len(s_list)
    
    if s_size == 1:
        pos[s_list[0]] = (0.0, 0.0)
    else:
        # Very small circle for S nodes at center - keep them close!
        center_radius = 0.07 + 0.05 * s_size  # Scales slightly with size but stays tight
        for i, node in enumerate(s_list):
            angle = 2 * np.pi * i / s_size + random.uniform(-0.1, 0.1)
            pos[node] = (center_radius * np.cos(angle), center_radius * np.sin(angle))
    
    # Place each component radially around the cut-set
    num_comp = len(components)
    
    # Base distance depends on number of components - closer if more components
    base_distance = 1.8 if num_comp <= 4 else 2.2
    
    for comp_idx, comp in enumerate(components):
        # Angle for this component (evenly distributed with slight randomness)
        sector_angle = 2 * np.pi * comp_idx / num_comp + random.uniform(-0.1, 0.1)
        
        # Distance from center - varies slightly for visual interest
        comp_distance = base_distance + random.uniform(-0.2, 0.3)
        comp_center_x = comp_distance * np.cos(sector_angle)
        comp_center_y = comp_distance * np.sin(sector_angle)
        
        comp_list = sorted(comp)
        n_comp = len(comp_list)
        
        if n_comp == 1:
            # Single node component
            pos[comp_list[0]] = (comp_center_x, comp_center_y)
        elif n_comp == 2:
            # Two nodes - place close together
            offset = 0.15
            pos[comp_list[0]] = (comp_center_x - offset, comp_center_y)
            pos[comp_list[1]] = (comp_center_x + offset, comp_center_y)
        else:
            # Use spring layout for this component subgraph
            # Scale based on component size - smaller scale = tighter cluster
            scale = min(0.6, 0.2 + 0.05 * n_comp)  # Tighter clusters
            subgraph = G.subgraph(comp)
            comp_pos = nx.spring_layout(
                subgraph,
                k=0.6,  # Smaller k = nodes closer together
                iterations=150,
                scale=scale,
                center=(comp_center_x, comp_center_y),
                seed=42 + comp_idx
            )
            pos.update(comp_pos)
    
    # Handle any remaining nodes not in components (shouldn't happen)
    for node in G.nodes():
        if node not in pos:
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(1.5, 2.0)
            pos[node] = (r * np.cos(angle), r * np.sin(angle))
    
    return pos, cut_set


# ============================================================================
# METADATA
# ============================================================================

def get_hamiltonian_metadata(G: nx.Graph, auxiliary, label: int) -> dict:
    """Generate metadata for Hamiltonian property."""
    metadata = {
        'property': 'hamiltonian',
        'label': label,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'ground_truth': {
            'has_hamiltonian_cycle': label == 1
        }
    }
    
    if auxiliary:
        if isinstance(auxiliary, list):
            # YES or s_no: Cycle (Hamiltonian for YES, longest for s_no)
            metadata['ground_truth']['cycle_length'] = len(auxiliary)
            metadata['ground_truth']['cycle_covers_all'] = len(auxiliary) == G.number_of_nodes()
        elif isinstance(auxiliary, tuple) and len(auxiliary) == 2:
            # d_no: (cut_set, components)
            cut_set, components = auxiliary
            metadata['ground_truth']['cut_set_size'] = len(cut_set)
            metadata['ground_truth']['num_components_when_removed'] = len(components)
            metadata['ground_truth']['violates_theorem'] = len(components) > len(cut_set)
            metadata['ground_truth']['is_partial'] = True  # d_no visualization is partial
    
    return metadata
