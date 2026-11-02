"""
Outer-Planarity Property Generator

Generates outer-planar/non-outer-planar graphs following GRAPHTRAILS Fig 8b/8d conventions:
- YES (Outer-Planar): All vertices on outer face, crossing-free drawing
- NO (Non-Outer-Planar): Contains K₄ or K₂,₃ minor highlighted

GRAPHTRAILS: "every vertex is located on the outer face" (line 803-804)

Generation Strategy v3 (Enhanced Randomization):
- Graphs have 12-50 nodes (wider range for unpredictability)
- Multiple outer-planar construction strategies
- Per-image unique seeding for randomness
- NO graph generated first, then s_no and d_no derived from SAME NO graph
- s_no: Uses same layout as YES (circular layout showing crossings)
- d_no: Uses minor visualization
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
MAX_NODES = 50  # Increased from 40


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_k4_edges() -> List[Tuple[int, int]]:
    """Return edges of K₄ (complete graph on 4 vertices)."""
    return [(i, j) for i in range(4) for j in range(i+1, 4)]


def _get_k23_edges() -> List[Tuple[int, int]]:
    """Return edges of K₂,₃ (complete bipartite 2,3)."""
    # Nodes 0,1 on one side, 2,3,4 on other
    return [(i, j) for i in range(2) for j in range(2, 5)]


def _is_outerplanar(G: nx.Graph) -> bool:
    """
    Check if graph is outerplanar using the correct characterization:
    A graph G is outerplanar iff adding a new vertex connected to ALL existing vertices
    results in a planar graph.
    
    This correctly detects K₄ and K₂,₃ minors/subdivisions.
    """
    if G.number_of_nodes() == 0:
        return True
    
    # Quick check: must be planar first
    if not nx.check_planarity(G)[0]:
        return False
    
    # Key characterization: G is outerplanar iff G + universal vertex is planar
    # Create a copy and add a universal vertex connected to all nodes
    H = G.copy()
    universal_node = max(G.nodes()) + 1 if G.nodes() else 0
    H.add_node(universal_node)
    for node in G.nodes():
        H.add_edge(universal_node, node)
    
    # If H is planar, then G is outerplanar
    return nx.check_planarity(H)[0]


def _find_maximal_outerplanar_subgraph(G: nx.Graph) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """
    Find a maximal outer-planar subgraph of G.
    
    Returns:
        outerplanar_subgraph: An outer-planar subgraph with as many edges as possible
        crossing_edges: Edges NOT in the subgraph (these will cross in circular layout)
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    
    edges = list(G.edges())
    random.shuffle(edges)
    
    crossing_edges = []
    
    for u, v in edges:
        H.add_edge(u, v)
        if not _is_outerplanar(H):
            H.remove_edge(u, v)
            crossing_edges.append((u, v))
    
    return H, crossing_edges


def _generate_circular_positions(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    """
    Generate circular positions for s_no layout using maximal outer-planar subgraph.
    
    Strategy:
    1. Find maximal outer-planar subgraph
    2. Order nodes using DFS on that subgraph (good circular ordering)
    3. Place on circle - the non-outer-planar edges will cross
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Find maximal outer-planar subgraph to get good node ordering
    outerplanar_subgraph, crossing_edges = _find_maximal_outerplanar_subgraph(G)
    
    # Use DFS ordering on the outer-planar subgraph for good circle placement
    if outerplanar_subgraph.number_of_edges() > 0 and nx.is_connected(outerplanar_subgraph):
        start_node = random.choice(list(outerplanar_subgraph.nodes()))
        ordered_nodes = list(nx.dfs_preorder_nodes(outerplanar_subgraph, start_node))
    else:
        ordered_nodes = nodes
    
    pos = {}
    angle_offset = random.uniform(0, 2 * np.pi)
    radius = 3.0  # Same radius as YES for consistency
    
    for i, node in enumerate(ordered_nodes):
        angle = angle_offset + 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[node] = (x, y)
    
    return pos


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_outer_planarity_graph_yes(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, None]:
    """
    Generate an OUTER-PLANAR graph with high randomization.
    
    GRAPHTRAILS Fig 8b: "every vertex is visible from above"
    
    Strategy: 
    - Use various outerplanar graph generators with randomization
    - Verify using proper outerplanarity test (G + universal vertex is planar)
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Outer-planar NetworkX graph
        None: No auxiliary data needed
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 50
    
    for _ in range(max_attempts):
        n = random.randint(min_nodes, max_nodes)
        
        # Strategy: Various outerplanar graph constructions with different densities
        graph_type = random.choices(
            ['cycle_with_chords', 'maximal_outerplanar', 'tree_plus', 
             'random_triangulation', 'caterpillar_plus', 'nested_cycles'],
            weights=[0.2, 0.15, 0.2, 0.2, 0.15, 0.1],
            k=1
        )[0]
        
        try:
            if graph_type == 'cycle_with_chords':
                # Start with cycle (outerplanar), add non-crossing chords
                G = nx.cycle_graph(n)
                # Randomize number of chords to add
                num_chords = random.randint(n // 6, n // 2)
                for _ in range(num_chords):
                    u, v = random.sample(range(n), 2)
                    if abs(u - v) > 2 and abs(u - v) < n - 2:
                        if not G.has_edge(u, v):
                            G.add_edge(u, v)
                            if not _is_outerplanar(G):
                                G.remove_edge(u, v)
                                
            elif graph_type == 'maximal_outerplanar':
                # Maximal outerplanar: triangulated polygon with random triangulation
                G = nx.cycle_graph(n)
                # Random triangulation by picking random vertices as fan centers
                nodes_list = list(range(n))
                random.shuffle(nodes_list)
                fan_centers = nodes_list[:random.randint(1, 3)]
                for center in fan_centers:
                    for i in range(n):
                        if i != center and (i + 1) % n != center and (i - 1) % n != center:
                            if not G.has_edge(center, i):
                                G.add_edge(center, i)
                                if not _is_outerplanar(G):
                                    G.remove_edge(center, i)
                        
            elif graph_type == 'random_triangulation':
                # Build triangulation incrementally by ear addition
                G = nx.cycle_graph(n)
                # Add random non-crossing chords to create triangles
                attempts = n * 2
                for _ in range(attempts):
                    u, v = random.sample(range(n), 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        if not _is_outerplanar(G):
                            G.remove_edge(u, v)
                            
            elif graph_type == 'caterpillar_plus':
                # Caterpillar graph (always outerplanar) with extra edges
                # Spine of the caterpillar
                spine_len = random.randint(n // 3, n // 2)
                G = nx.path_graph(spine_len)
                node_counter = spine_len
                # Add leaves to spine nodes
                while node_counter < n:
                    spine_node = random.randint(0, spine_len - 1)
                    G.add_edge(spine_node, node_counter)
                    node_counter += 1
                # Add extra edges while keeping outerplanar
                for _ in range(n // 3):
                    u, v = random.sample(list(G.nodes()), 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        if not _is_outerplanar(G):
                            G.remove_edge(u, v)
                            
            elif graph_type == 'nested_cycles':
                # Create nested structure with connecting edges
                G = nx.Graph()
                G.add_nodes_from(range(n))
                # Create outer cycle
                for i in range(n):
                    G.add_edge(i, (i + 1) % n)
                # Add random skip edges (chords)
                skip_amounts = [2, 3, random.randint(4, max(5, n // 4))]
                for skip in skip_amounts:
                    if skip < n // 2:
                        for i in range(0, n, random.randint(2, 4)):
                            j = (i + skip) % n
                            if not G.has_edge(i, j):
                                G.add_edge(i, j)
                                if not _is_outerplanar(G):
                                    G.remove_edge(i, j)
                        
            else:  # tree_plus
                # Tree (always outerplanar) plus edges
                G = nx.random_tree(n)
                # Add more edges while keeping outerplanar
                num_extra = random.randint(n // 4, n // 2)
                for _ in range(num_extra):
                    u, v = random.sample(list(G.nodes()), 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        if not _is_outerplanar(G):
                            G.remove_edge(u, v)
            
            # Verify it's outerplanar and connected
            if _is_outerplanar(G) and nx.is_connected(G):
                # Randomly relabel nodes for additional variety
                mapping = {old: new for new, old in enumerate(random.sample(list(G.nodes()), len(G.nodes())))}
                G = nx.relabel_nodes(G, mapping)
                return G, None
                
        except Exception:
            continue
    
    # Fallback: simple cycle with some chords
    n = random.randint(min_nodes, max_nodes)
    G = nx.cycle_graph(n)
    for i in range(2, n - 1, 2):
        G.add_edge(0, i)
    return G, None


def generate_outer_planarity_graph_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a NON-OUTER-PLANAR graph containing K₄ or K₂,₃ as a MINOR or SUBDIVISION.
    This is the UNIFIED generator - s_no and d_no are derived from the same graph.
    
    GRAPHTRAILS Fig 8d: "K₂,₃ is highlighted in a non-planar drawing"
    
    A graph is outerplanar iff it does not contain K₄ or K₂,₃ as a minor.
    (Note: Graph can be planar but not outer-planar)
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-outer-planar NetworkX graph (may still be planar)
        info: Dict containing:
            - 'minor_info': (minor_type, minor_nodes, minor_edges, edge_paths) for d_no layout
            - 'template_pos': Circular positions for s_no layout (showing crossings)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 50
    
    for _ in range(max_attempts):
        # Randomly choose K₄ or K₂,₃
        minor_type = random.choice(['K4', 'K23'])
        
        # Decide whether to create subdivision or minor (both valid)
        use_subdivision = random.choice([True, False])
        
        if minor_type == 'K4':
            if use_subdivision:
                # Create SUBDIVISION of K₄ (add vertices along edges)
                G = nx.Graph()
                original_nodes = list(range(4))
                G.add_nodes_from(original_nodes)
                
                node_counter = 4
                minor_nodes = list(range(4))  # Branch vertices (core K4 nodes)
                subdivision_nodes = []  # All subdivision nodes on paths
                minor_edges = set()  # Actual edges in the graph that form subdivision
                edge_paths = {}  # Maps (branch_u, branch_v) -> list of nodes in path
                
                # For each edge of K₄, randomly subdivide it
                for i in range(4):
                    for j in range(i+1, 4):
                        # Subdivide edge with 0-2 intermediate nodes
                        num_subdivisions = random.randint(0, 2)
                        if num_subdivisions == 0:
                            G.add_edge(i, j)
                            minor_edges.add((i, j))
                            edge_paths[(i, j)] = []  # No intermediate nodes
                        else:
                            # Create path from i to j through subdivision nodes
                            path_nodes = []
                            prev = i
                            for k in range(num_subdivisions):
                                G.add_edge(prev, node_counter)
                                minor_edges.add((prev, node_counter))  # Add actual edge
                                subdivision_nodes.append(node_counter)
                                path_nodes.append(node_counter)
                                prev = node_counter
                                node_counter += 1
                            G.add_edge(prev, j)
                            minor_edges.add((prev, j))  # Add final edge
                            edge_paths[(i, j)] = path_nodes
                
                # All nodes that form the subdivision (branch vertices + subdivision nodes)
                all_minor_nodes = minor_nodes + subdivision_nodes
                base_size = node_counter
            else:
                # Create MINOR of K₄ (start with K₄)
                G = nx.complete_graph(4)
                minor_nodes = list(range(4))
                all_minor_nodes = minor_nodes.copy()
                minor_edges = set(_get_k4_edges())  # Actual edges
                edge_paths = {edge: [] for edge in _get_k4_edges()}  # No subdivisions
                base_size = 4
        else:  # K23
            if use_subdivision:
                # Create SUBDIVISION of K₂,₃
                G = nx.Graph()
                original_nodes = list(range(5))  # 0,1 on left, 2,3,4 on right
                G.add_nodes_from(original_nodes)
                
                node_counter = 5
                minor_nodes = list(range(5))  # Branch vertices (core K23 nodes)
                subdivision_nodes = []  # All subdivision nodes on paths
                minor_edges = set()  # Actual edges in the graph that form subdivision
                edge_paths = {}  # Maps (branch_u, branch_v) -> list of nodes in path
                
                # For each edge of K₂,₃, randomly subdivide it
                for i in range(2):
                    for j in range(2, 5):
                        # Subdivide edge with 0-2 intermediate nodes
                        num_subdivisions = random.randint(0, 2)
                        if num_subdivisions == 0:
                            G.add_edge(i, j)
                            minor_edges.add((i, j))
                            edge_paths[(i, j)] = []  # No intermediate nodes
                        else:
                            # Create path from i to j through subdivision nodes
                            path_nodes = []
                            prev = i
                            for k in range(num_subdivisions):
                                G.add_edge(prev, node_counter)
                                minor_edges.add((prev, node_counter))  # Add actual edge
                                subdivision_nodes.append(node_counter)
                                path_nodes.append(node_counter)
                                prev = node_counter
                                node_counter += 1
                            G.add_edge(prev, j)
                            minor_edges.add((prev, j))  # Add final edge
                            edge_paths[(i, j)] = path_nodes
                
                # All nodes that form the subdivision (branch vertices + subdivision nodes)
                all_minor_nodes = minor_nodes + subdivision_nodes
                base_size = node_counter
            else:
                # Create MINOR of K₂,₃ (start with K₂,₃)
                G = nx.complete_bipartite_graph(2, 3)
                minor_nodes = list(range(5))
                all_minor_nodes = minor_nodes.copy()
                minor_edges = set(_get_k23_edges())  # Actual edges
                edge_paths = {edge: [] for edge in _get_k23_edges()}  # No subdivisions
                base_size = 5
        
        # Add extra nodes to reach desired size (keep sparse to potentially stay planar)
        n = random.randint(min_nodes, max_nodes)
        extra_nodes = max(0, n - base_size)
        
        for i in range(extra_nodes):
            new_node = base_size + i
            G.add_node(new_node)
            # Connect to 1-3 random nodes (keep sparse to stay planar if possible)
            num_connections = random.randint(1, 3)
            available = [n for n in G.nodes() if n != new_node]
            targets = random.sample(available, min(num_connections, len(available)))
            
            for t in targets:
                G.add_edge(new_node, t)
        
        # Verify not outerplanar and connected
        if not _is_outerplanar(G) and nx.is_connected(G):
            minor_info = (minor_type, all_minor_nodes, minor_edges, edge_paths)
            # Generate template positions for s_no (circular layout showing crossings)
            template_pos = _generate_circular_positions(G)
            return G, {'minor_info': minor_info, 'template_pos': template_pos}
    
    # Fallback: Simple K₄ subdivision
    G = nx.Graph()
    # K4 vertices: 0,1,2,3
    # Subdivide each edge with one extra node
    node_counter = 4
    minor_nodes = list(range(4))
    all_minor_nodes = list(range(4))  # Will include subdivision nodes
    minor_edges = set()  # Actual edges
    edge_paths = {}  # Maps (i, j) -> [subdivision_node]
    for i in range(4):
        for j in range(i+1, 4):
            # Add subdivision node
            G.add_edge(i, node_counter)
            G.add_edge(node_counter, j)
            minor_edges.add((i, node_counter))  # Add actual edge
            minor_edges.add((node_counter, j))  # Add actual edge
            edge_paths[(i, j)] = [node_counter]
            all_minor_nodes.append(node_counter)
            node_counter += 1
    
    minor_info = ('K4', all_minor_nodes, minor_edges, edge_paths)
    template_pos = _generate_circular_positions(G)
    return G, {'minor_info': minor_info, 'template_pos': template_pos}


def generate_outer_planarity_graph_s_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate an s_no graph: Uses the SAME layout as YES (circular layout showing crossings).
    Derived from the unified NO generator.

    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-outerplanar NetworkX graph
        template_pos: Circular positions for layout (will show crossings)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract s_no data
    G, info = generate_outer_planarity_graph_no(min_nodes, max_nodes)
    return G, info['template_pos']


def generate_outer_planarity_graph_d_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Tuple]:
    """
    Generate a d_no graph: Uses minor visualization.
    Derived from the unified NO generator.

    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-outerplanar NetworkX graph
        minor_info: (minor_type, minor_nodes, minor_edges, edge_paths) for layout
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract d_no data
    G, info = generate_outer_planarity_graph_no(min_nodes, max_nodes)
    return G, info['minor_info']


# ============================================================================
# LAYOUT FUNCTIONS
# ============================================================================

def _get_outer_face_ordering(G: nx.Graph) -> List:
    """
    Get the ordering of vertices on the outer face of an outer-planar graph.
    This ordering ensures NO edge crossings when placed on a circle.
    
    Uses the fact that for outer-planar graphs, adding a universal vertex
    creates a planar graph, and the outer face of G corresponds to the
    face containing the universal vertex.
    """
    if G.number_of_nodes() == 0:
        return []
    
    if G.number_of_nodes() == 1:
        return list(G.nodes())
    
    if G.number_of_nodes() == 2:
        return list(G.nodes())
    
    # Create augmented graph with universal vertex
    H = G.copy()
    universal = max(G.nodes()) + 1
    H.add_node(universal)
    for node in G.nodes():
        H.add_edge(universal, node)
    
    # Get planar embedding
    is_planar, embedding = nx.check_planarity(H)
    if not is_planar:
        # Fallback: just return nodes in some order
        return list(G.nodes())
    
    # The outer face of G is the face in H that contains the universal vertex
    # In the embedding, find the face around universal vertex
    # The neighbors of universal in the embedding give the outer face order
    try:
        # Get the cyclic order of neighbors around universal vertex
        neighbors_cw = list(embedding.neighbors_cw_order(universal))
        # This is the outer face ordering (may need to reverse for correct orientation)
        return neighbors_cw
    except Exception:
        # Fallback
        return list(G.nodes())


def layout_outer_planarity_yes(G: nx.Graph, auxiliary) -> Dict[int, Tuple[float, float]]:
    """
    GRAPHTRAILS Fig 8b layout for OUTER-PLANAR graphs.
    
    Strategy: All vertices on outer boundary using PLANAR EMBEDDING to get
    the correct outer face ordering - this guarantees NO edge crossings.
    
    Visual proof: All nodes on outer circle, crossing-free.
    
    For s_no: If auxiliary is a template_pos dict, use it (will show crossings).
    """
    # If auxiliary is a precomputed position template (from s_no), use it directly
    if auxiliary and isinstance(auxiliary, dict) and all(isinstance(v, tuple) for v in auxiliary.values()):
        pos = {n: auxiliary[n] for n in G.nodes() if n in auxiliary}
        return pos
    
    n = G.number_of_nodes()
    
    # Get proper outer face ordering using planar embedding
    # This guarantees NO crossings for outer-planar graphs
    ordered_nodes = _get_outer_face_ordering(G)
    
    # Ensure all nodes are included (defensive)
    if set(ordered_nodes) != set(G.nodes()):
        ordered_nodes = list(G.nodes())
    
    pos = {}
    # Random starting angle for variety
    angle_offset = random.uniform(0, 2 * np.pi)
    radius = 3.0  # Larger radius for better spread
    
    # Place nodes on outer circle in outer-face order (NO crossings)
    for i, node in enumerate(ordered_nodes):
        angle = angle_offset + 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[node] = (x, y)
    
    return pos


def _k4_layout() -> Dict[int, Tuple[float, float]]:
    """Square/diamond layout for K₄ minor nodes with randomized size and rotation."""
    radius = random.uniform(1.0, 2.0)
    rot_offset = random.uniform(0, 2 * np.pi)
    pos = {}
    for i in range(4):
        angle = np.pi / 2 * i + np.pi / 2 + rot_offset
        pos[i] = (radius * np.cos(angle), radius * np.sin(angle))
    return pos


def _k23_layout() -> Dict[int, Tuple[float, float]]:
    """Bipartite layout for K₂,₃ minor nodes with randomized spacing."""
    x_dist = random.uniform(1.0, 2.2)
    y_span_left = random.uniform(0.6, 1.2)
    y_span_right = random.uniform(0.8, 1.6)
    orientation = random.choice(['horizontal', 'vertical'])
    if orientation == 'horizontal':
        return {
            0: (-x_dist, y_span_left),
            1: (-x_dist, -y_span_left),
            2: (x_dist, y_span_right),
            3: (x_dist, 0),
            4: (x_dist, -y_span_right)
        }
    else:
        return {
            0: (y_span_left, -x_dist),
            1: (-y_span_left, -x_dist),
            2: (y_span_right, x_dist),
            3: (0, x_dist),
            4: (-y_span_right, x_dist)
        }


def layout_outer_planarity_no(G: nx.Graph, minor_info: Tuple) -> Dict[int, Tuple[float, float]]:
    """
    GRAPHTRAILS Fig 8d layout for NON-OUTER-PLANAR graphs.
    
    Strategy:
    1. Place K₄ or K₂,₃ branch vertices in fixed recognizable layout
    2. Place subdivision nodes on straight lines between branch vertices
    3. Place other nodes around using spring layout
    
    Visual proof: The highlighted minor shows why not outer-planar.
    """
    if len(minor_info) == 4:
        minor_type, minor_nodes, minor_edges, edge_paths = minor_info
    else:
        # Backward compatibility if edge_paths not provided
        minor_type, minor_nodes, minor_edges = minor_info
        edge_paths = {}
    
    # Branch vertices are the first 4 (K4) or 5 (K23) nodes
    if minor_type == 'K4':
        branch_vertices = [n for n in minor_nodes if n < 4]
        minor_pos = _k4_layout()
    else:  # K23
        branch_vertices = [n for n in minor_nodes if n < 5]
        minor_pos = _k23_layout()
    
    other_nodes = [n for n in G.nodes() if n not in minor_nodes]
    
    pos = {}
    
    # Place branch vertices in fixed layout
    for node in branch_vertices:
        if node in minor_pos:
            pos[node] = minor_pos[node]
    
    # Place subdivision nodes on straight lines between branch vertices
    for (u, v), path_nodes in edge_paths.items():
        if u in pos and v in pos and path_nodes:
            # Interpolate positions along straight line from u to v
            u_pos = np.array(pos[u])
            v_pos = np.array(pos[v])
            num_nodes = len(path_nodes)
            for i, node in enumerate(path_nodes):
                # Position at (i+1)/(num_nodes+1) along the line from u to v
                t = (i + 1) / (num_nodes + 1)
                node_pos = u_pos + t * (v_pos - u_pos)
                pos[node] = tuple(node_pos)
    
    # Place other nodes around the minor
    if other_nodes:
        for i, node in enumerate(other_nodes):
            angle = 2 * np.pi * i / len(other_nodes) + np.pi / 4
            r = 2.8
            pos[node] = (r * np.cos(angle), r * np.sin(angle))
    
    # Determine bold edges corresponding to the minor (K4 or K2,3)
    bold_edges = set()
    try:
        # minor_edges is a set of edge tuples collected during generation
        if minor_edges:
            # Keep only edges that actually exist in G (defensive)
            bold_edges = set((u, v) for (u, v) in minor_edges if G.has_edge(u, v))
    except Exception:
        bold_edges = set()

    return pos, bold_edges


def get_outer_planarity_metadata(G: nx.Graph, auxiliary, label: int) -> dict:
    """Generate metadata for outer-planarity property."""
    is_outerplanar = _is_outerplanar(G)
    is_planar = nx.check_planarity(G)[0]
    
    metadata = {
        'property': 'outer_planarity',
        'label': label,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'ground_truth': {
            'is_outerplanar': is_outerplanar,
            'is_planar': is_planar
        }
    }
    
    if label == 0 and auxiliary:  # Non-outer-planar with minor
        # Auxiliary may be tuple/list (minor info) or other types; handle safely
        if isinstance(auxiliary, (tuple, list)):
            if len(auxiliary) == 4:
                minor_type, minor_nodes, minor_edges, edge_paths = auxiliary
            elif len(auxiliary) == 3:
                minor_type, minor_nodes, minor_edges = auxiliary
            else:
                minor_type = None
                minor_nodes = []

            if minor_type is not None:
                metadata['ground_truth']['minor_type'] = minor_type
                metadata['ground_truth']['minor_size'] = len(minor_nodes)

    return metadata
