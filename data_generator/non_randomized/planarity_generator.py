"""
Planarity Property Generator

Generates planar/non-planar graphs following GRAPHTRAILS Fig 8 conventions:
- YES (Planar): Crossing-free embedding
- NO (Non-Planar): K₅ or K₃,₃ minor highlighted

Generation Strategy v2:
- All graphs generated with 20-40 nodes (more randomized)
- NO graph generated first, then s_no and d_no derived from SAME NO graph
- s_no: Uses same layout as YES (planar layout) showing crossings
- d_no: Uses Kuratowski minor visualization
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
# HELPER FUNCTIONS
# ============================================================================

def _is_planar(G: nx.Graph) -> bool:
    """
    Check if graph is planar using NetworkX's proper planarity test.
    NetworkX uses the Booth-Lueker or Boyer-Myrvold algorithm which correctly
    detects K₅ and K₃,₃ minors.
    
    This is the gold standard for planarity testing.
    """
    if G.number_of_nodes() == 0:
        return True
    
    # NetworkX's check_planarity uses proper planarity testing algorithms
    # (Booth-Lueker or Boyer-Myrvold) that detect K5/K3,3 minors correctly
    return nx.check_planarity(G)[0]


def _get_k5_edges() -> List[Tuple[int, int]]:
    """Return edges of K₅ (complete graph on 5 vertices)."""
    return [(i, j) for i in range(5) for j in range(i+1, 5)]


def _get_k33_edges() -> List[Tuple[int, int]]:
    """Return edges of K₃,₃ (complete bipartite 3,3)."""
    # Nodes 0,1,2 on one side, 3,4,5 on other
    return [(i, j) for i in range(3) for j in range(3, 6)]


def _k5_layout() -> Dict[int, Tuple[float, float]]:
    """Pentagon layout for K₅ minor nodes."""
    pos = {}
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi / 2  # Start from top
        pos[i] = (1.8 * np.cos(angle), 1.8 * np.sin(angle))
    return pos


def _k33_layout() -> Dict[int, Tuple[float, float]]:
    """Bipartite layout for K₃,₃ minor nodes."""
    pos = {}
    # Left side (nodes 0, 1, 2)
    for i in range(3):
        pos[i] = (-1.5, 1.5 - i * 1.5)
    # Right side (nodes 3, 4, 5)
    for i in range(3, 6):
        pos[i] = (1.5, 1.5 - (i - 3) * 1.5)
    return pos


def _find_maximal_planar_subgraph(G: nx.Graph) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """
    Find a maximal planar subgraph of G.
    
    Returns:
        planar_subgraph: A planar subgraph of G containing as many edges as possible
        crossing_edges: List of edges NOT in the planar subgraph (these will cross)
    """
    # Start with all nodes, no edges
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    
    # Shuffle edges for randomness
    edges = list(G.edges())
    random.shuffle(edges)
    
    crossing_edges = []
    
    # Greedily add edges that keep the graph planar
    for u, v in edges:
        H.add_edge(u, v)
        is_planar, _ = nx.check_planarity(H)
        if not is_planar:
            H.remove_edge(u, v)
            crossing_edges.append((u, v))
    
    return H, crossing_edges


def _generate_template_positions(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    """
    Generate positions for s_no layout using maximal planar subgraph approach.
    
    Strategy:
    1. Find maximal planar subgraph (subset of edges that are planar)
    2. Compute planar layout for that subgraph
    3. Use those positions - the remaining edges will naturally cross
    
    This ensures s_no uses "planar-style" layout but shows crossings for non-planar edges.
    """
    # Find maximal planar subgraph
    planar_subgraph, crossing_edges = _find_maximal_planar_subgraph(G)
    
    # Compute planar layout for the planar subgraph
    try:
        is_planar, embedding = nx.check_planarity(planar_subgraph)
        if is_planar:
            pos = nx.planar_layout(planar_subgraph, scale=5.0)  # Larger scale
        else:
            # Shouldn't happen, but fallback
            pos = nx.spring_layout(planar_subgraph, k=5.0, iterations=300, scale=5.0)
    except Exception:
        pos = nx.spring_layout(planar_subgraph, k=4.0, iterations=200, scale=5.0)
    
    # Spread nodes from centroid for better visibility
    coords = np.array(list(pos.values()))
    if len(coords) > 0:
        center = coords.mean(axis=0)
        spread = 2.5  # Increased spread for better spacing
        for node, (x, y) in list(pos.items()):
            vec = np.array([x, y]) - center
            pos[node] = tuple(center + vec * spread)
    
    return pos


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_planarity_graph_yes(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, None]:
    """
    Generate a PLANAR graph.
    
    GRAPHTRAILS: "Crossing-free embedding proves planarity"
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Planar NetworkX graph
        None: No auxiliary data needed
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 50
    
    for _ in range(max_attempts):
        n = random.randint(min_nodes, max_nodes)
        
        # Strategy: Use various planar graph generators with more randomness
        # Weighted choice - prefer denser planar graphs
        graph_type = random.choices(
            ['tree', 'maximal_planar', 'grid', 'wheel', 'triangulation_insert', 'random_edge_add', 'delaunay_like'],
            weights=[0.1, 0.25, 0.1, 0.1, 0.2, 0.15, 0.1],
            k=1
        )[0]
        
        try:
            if graph_type == 'tree':
                G = nx.random_tree(n)
                # Add more edges while keeping planar
                nodes = list(G.nodes())
                target_edges = random.randint(n, int(n * 1.5))
                for _ in range(n * 3):
                    if G.number_of_edges() >= target_edges:
                        break
                    u, v = random.sample(nodes, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        is_planar, _ = nx.check_planarity(G)
                        if not is_planar:
                            G.remove_edge(u, v)
                                
            elif graph_type == 'maximal_planar':
                # Start with triangle and insert nodes into random existing triangles
                G = nx.cycle_graph(3)
                node_counter = 3
                while node_counter < n:
                    # Find triangles (faces) in G
                    triangles = set()
                    for u in G.nodes():
                        for v in G.neighbors(u):
                            for w in G.neighbors(v):
                                if w in G[u] and u < v < w:
                                    triangles.add((u, v, w))
                    if triangles:
                        tri = random.choice(list(triangles))
                        G.add_node(node_counter)
                        for t in tri:
                            G.add_edge(node_counter, t)
                        node_counter += 1
                    else:
                        # No triangle to insert into; fall back to safe edge additions
                        break
                
            elif graph_type == 'triangulation_insert':
                # Aggressive triangulation: always insert into triangles to produce a maximal planar graph
                G = nx.cycle_graph(3)
                node_counter = 3
                while node_counter < n:
                    # Collect unique triangles
                    triangles = set()
                    for u in G.nodes():
                        for v in G.neighbors(u):
                            for w in G.neighbors(v):
                                if w in G[u]:
                                    tri = tuple(sorted((u, v, w)))
                                    triangles.add(tri)
                    if triangles:
                        tri = random.choice(list(triangles))
                        G.add_node(node_counter)
                        for t in tri:
                            G.add_edge(node_counter, t)
                        node_counter += 1
                    else:
                        break

            elif graph_type == 'random_edge_add':
                # Start with a planar base and randomly add edges that keep planarity
                G = nx.random_tree(n)
                nodes = list(G.nodes())
                for _ in range(n * 3):
                    u, v = random.sample(nodes, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        is_planar, _ = nx.check_planarity(G)
                        if not is_planar:
                            G.remove_edge(u, v)

            elif graph_type == 'grid':
                # Approximate grid with some randomness
                rows = random.randint(3, 6)
                cols = max(3, n // rows)
                G = nx.grid_2d_graph(rows, cols)
                G = nx.convert_node_labels_to_integers(G)
                # Randomly add edges preserving planarity for more variety
                nodes = list(G.nodes())
                for _ in range(max(1, n // 3)):
                    u, v = random.sample(nodes, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        is_planar, _ = nx.check_planarity(G)
                        if not is_planar:
                            G.remove_edge(u, v)
            
            elif graph_type == 'delaunay_like':
                # Create Delaunay-like triangulation from random points
                # This creates organic-looking planar graphs
                points = np.random.rand(n, 2) * 10  # Random points in [0, 10] x [0, 10]
                from scipy.spatial import Delaunay
                try:
                    tri = Delaunay(points)
                    G = nx.Graph()
                    G.add_nodes_from(range(n))
                    for simplex in tri.simplices:
                        for i in range(3):
                            for j in range(i+1, 3):
                                G.add_edge(simplex[i], simplex[j])
                except Exception:
                    # Fallback to random tree
                    G = nx.random_tree(n)
                
            else:  # wheel
                G = nx.wheel_graph(n)
                # Randomly add cross chords if planar
                rim = list(range(1, n))
                for _ in range(max(1, n // 4)):
                    u, v = random.sample(rim, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        is_planar, _ = nx.check_planarity(G)
                        if not is_planar:
                            G.remove_edge(u, v)
            
            if _is_planar(G) and nx.is_connected(G):
                # NetworkX's planarity test is reliable - if it says planar, it is planar
                # Relabel nodes randomly for additional randomness in layouts
                mapping = {old: new for new, old in enumerate(random.sample(list(G.nodes()), len(G.nodes())))}
                G = nx.relabel_nodes(G, mapping)
                return G, None
                
        except Exception:
            continue
    
    # Fallback: simple tree
    G = nx.random_tree(random.randint(10, 15))
    return G, None


def generate_planarity_graph_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a NON-PLANAR graph containing K₅ or K₃,₃ as a MINOR or SUBDIVISION.
    This is the UNIFIED generator - s_no and d_no are derived from the same graph.
    
    GRAPHTRAILS Fig 8c: "Highlight the Kuratowski subdivision"
    
    Kuratowski's Theorem: A graph is planar iff it contains no subdivision of K₅ or K₃,₃
    Wagner's Theorem: A graph is planar iff it contains no K₅ or K₃,₃ minor
    
    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-planar NetworkX graph
        info: Dict containing:
            - 'minor_info': (minor_type, minor_nodes, minor_edges, edge_paths) for d_no layout
            - 'template_pos': Planar-style positions for s_no layout (showing crossings)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    max_attempts = 50
    
    for _ in range(max_attempts):
        # Randomly choose K₅ or K₃,₃
        minor_type = random.choice(['K5', 'K33'])
        
        # Decide whether to create subdivision or minor (both valid)
        use_subdivision = random.choice([True, False])
        
        if minor_type == 'K5':
            if use_subdivision:
                # Create SUBDIVISION of K₅ (add vertices along edges)
                G = nx.Graph()
                original_nodes = list(range(5))
                G.add_nodes_from(original_nodes)
                
                node_counter = 5
                minor_nodes = list(range(5))  # Branch vertices (core K5 nodes)
                subdivision_nodes = []  # All subdivision nodes on paths
                minor_edges = set()  # Actual edges in the graph that form subdivision
                edge_paths = {}  # Maps (branch_u, branch_v) -> list of nodes in path
                
                # For each edge of K₅, randomly subdivide it
                for i in range(5):
                    for j in range(i+1, 5):
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
                # Create MINOR of K₅ (start with K₅)
                G = nx.complete_graph(5)
                minor_nodes = list(range(5))
                all_minor_nodes = minor_nodes.copy()
                minor_edges = set(_get_k5_edges())  # Actual edges
                edge_paths = {edge: [] for edge in _get_k5_edges()}  # No subdivisions
                base_size = 5
        else:  # K33
            if use_subdivision:
                # Create SUBDIVISION of K₃,₃
                G = nx.Graph()
                original_nodes = list(range(6))  # 0,1,2 on left, 3,4,5 on right
                G.add_nodes_from(original_nodes)
                
                node_counter = 6
                minor_nodes = list(range(6))  # Branch vertices (core K33 nodes)
                subdivision_nodes = []  # All subdivision nodes on paths
                minor_edges = set()  # Actual edges in the graph that form subdivision
                edge_paths = {}  # Maps (branch_u, branch_v) -> list of nodes in path
                
                # For each edge of K₃,₃, randomly subdivide it
                for i in range(3):
                    for j in range(3, 6):
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
                # Create MINOR of K₃,₃ (start with K₃,₃)
                G = nx.complete_bipartite_graph(3, 3)
                minor_nodes = list(range(6))
                all_minor_nodes = minor_nodes.copy()
                minor_edges = set(_get_k33_edges())  # Actual edges
                edge_paths = {edge: [] for edge in _get_k33_edges()}  # No subdivisions
                base_size = 6
        
        # Add extra nodes to reach desired size
        n = random.randint(min_nodes, max_nodes)
        extra_nodes = max(0, n - base_size)
        
        for i in range(extra_nodes):
            new_node = base_size + i
            G.add_node(new_node)
            # Connect to 2-4 random nodes (preferring original minor nodes)
            num_connections = random.randint(2, 4)
            if random.random() < 0.7 and len(minor_nodes) > 0:
                targets = random.sample(minor_nodes, min(num_connections, len(minor_nodes)))
            else:
                available = [n for n in G.nodes() if n != new_node]
                targets = random.sample(available, min(num_connections, len(available)))
            
            for t in targets:
                G.add_edge(new_node, t)
        
        # Verify non-planar and connected
        is_planar, _ = nx.check_planarity(G)
        if not is_planar and nx.is_connected(G):
            minor_info = (minor_type, all_minor_nodes, minor_edges, edge_paths)
            # Generate template positions for s_no (planar-style layout showing crossings)
            template_pos = _generate_template_positions(G)
            return G, {'minor_info': minor_info, 'template_pos': template_pos}
    
    # Fallback: Simple K₅ subdivision
    G = nx.Graph()
    # K5 vertices: 0,1,2,3,4
    # Subdivide each edge with one extra node
    node_counter = 5
    minor_nodes = list(range(5))
    all_minor_nodes = list(range(5))  # Will include subdivision nodes
    minor_edges = set()  # Actual edges
    edge_paths = {}  # Maps (i, j) -> [subdivision_node]
    for i in range(5):
        for j in range(i+1, 5):
            # Add subdivision node
            G.add_edge(i, node_counter)
            G.add_edge(node_counter, j)
            minor_edges.add((i, node_counter))  # Add actual edge
            minor_edges.add((node_counter, j))  # Add actual edge
            edge_paths[(i, j)] = [node_counter]
            all_minor_nodes.append(node_counter)
            node_counter += 1
    
    minor_info = ('K5', all_minor_nodes, minor_edges, edge_paths)
    template_pos = _generate_template_positions(G)
    return G, {'minor_info': minor_info, 'template_pos': template_pos}


def generate_planarity_graph_s_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate an s_no graph: Uses the SAME layout as YES (planar layout showing crossings).
    Derived from the unified NO generator.

    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-planar NetworkX graph
        template_pos: Planar-style positions for layout (will show crossings)
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract s_no data
    G, info = generate_planarity_graph_no(min_nodes, max_nodes)
    return G, info['template_pos']


def generate_planarity_graph_d_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Tuple]:
    """
    Generate a d_no graph: Uses Kuratowski minor visualization.
    Derived from the unified NO generator.

    Args:
        min_nodes: Minimum number of nodes (defaults to global MIN_NODES)
        max_nodes: Maximum number of nodes (defaults to global MAX_NODES)
    
    Returns:
        G: Non-planar NetworkX graph
        minor_info: (minor_type, minor_nodes, minor_edges, edge_paths) for layout
    """
    if min_nodes is None:
        min_nodes = MIN_NODES
    if max_nodes is None:
        max_nodes = MAX_NODES
        
    # Call unified NO generator and extract d_no data
    G, info = generate_planarity_graph_no(min_nodes, max_nodes)
    return G, info['minor_info']


# ============================================================================
# LAYOUT FUNCTIONS
# ============================================================================

def layout_planarity_yes(G: nx.Graph, auxiliary) -> Dict[int, Tuple[float, float]]:
    """
    GRAPHTRAILS Fig 8a layout for PLANAR graphs.

    Strategy: Use planar embedding for crossing-free drawing and make the
    visualization more sparse by scaling and spreading nodes from the centroid.

    Visual proof: No edge crossings proves planarity; sparser spacing improves readability.

    Special behavior: If `auxiliary` is a template position mapping (from s_no generator),
    return that mapping unchanged so s_no images use the YES layout but show crossing edges.
    """
    # If auxiliary is a precomputed position template (node -> (x,y)), use it directly
    # This is used for s_no to ensure same layout style as YES
    if auxiliary and isinstance(auxiliary, dict) and all(isinstance(v, tuple) for v in auxiliary.values()):
        # Only keep positions for nodes that exist in G (defensive)
        pos = {n: auxiliary[n] for n in G.nodes() if n in auxiliary}
        return pos

    try:
        # Try planar layout first for planar graphs
        is_planar, embedding = nx.check_planarity(G)
        if is_planar:
            pos = nx.planar_layout(G, scale=5.0)  # Larger base scale

            # Spread nodes MORE away from centroid for better visibility
            coords = np.array(list(pos.values()))
            if len(coords) > 0:
                center = coords.mean(axis=0)
                spread = 2.5  # Increased spread for better spacing
                for node, (x, y) in list(pos.items()):
                    vec = np.array([x, y]) - center
                    pos[node] = tuple(center + vec * spread)

            return pos
    except Exception:
        pass

    # Fallback to spring layout with larger spring constant and more iterations
    pos = nx.spring_layout(G, k=5.0, iterations=400, scale=5.0, seed=random.randint(1, 10000))

    # Also spread nodes for readability
    coords = np.array(list(pos.values()))
    if len(coords) > 0:
        center = coords.mean(axis=0)
        spread = 2.5  # Increased spread for better spacing
        for node, (x, y) in list(pos.items()):
            vec = np.array([x, y]) - center
            pos[node] = tuple(center + vec * spread)

    return pos


def layout_planarity_no(G: nx.Graph, minor_info: Tuple) -> Dict[int, Tuple[float, float]]:
    """
    GRAPHTRAILS Fig 8c layout for NON-PLANAR graphs.
    
    Strategy:
    1. Place K₅ or K₃,₃ branch vertices in fixed recognizable layout
    2. Place subdivision nodes on straight lines between branch vertices
    3. Place other nodes around using spring layout
    
    Visual proof: The highlighted minor clearly shows non-planarity.
    """
    if len(minor_info) == 4:
        minor_type, minor_nodes, minor_edges, edge_paths = minor_info
    else:
        # Backward compatibility if edge_paths not provided
        minor_type, minor_nodes, minor_edges = minor_info
        edge_paths = {}
    
    # Branch vertices are the first 5 (K5) or 6 (K33) nodes
    if minor_type == 'K5':
        branch_vertices = [n for n in minor_nodes if n < 5]
        minor_pos = _k5_layout()
    else:  # K33
        branch_vertices = [n for n in minor_nodes if n < 6]
        minor_pos = _k33_layout()
    
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
    
    # Place other nodes using spring layout around the minor
    if other_nodes:
        # Create subgraph with other nodes and their connections
        other_subgraph = G.subgraph(other_nodes).copy()
        
        if len(other_nodes) > 1:
            other_pos = nx.spring_layout(other_subgraph, k=0.8, iterations=50, scale=1.0)
            
            # Offset to place around the minor
            for node, (x, y) in other_pos.items():
                # Place in outer ring
                angle = np.arctan2(y, x) if (x != 0 or y != 0) else random.uniform(0, 2 * np.pi)
                r = 2.8 + abs(x) * 0.5
                pos[node] = (r * np.cos(angle), r * np.sin(angle))
        else:
            # Single node: place randomly in outer area
            for node in other_nodes:
                angle = random.uniform(0, 2 * np.pi)
                pos[node] = (2.8 * np.cos(angle), 2.8 * np.sin(angle))
    
    # Build bold edges set from minor_edges to highlight the Kuratowski minor
    bold_edges = set()
    try:
        if minor_edges:
            bold_edges = set((u, v) for (u, v) in minor_edges if G.has_edge(u, v))
    except Exception:
        bold_edges = set()

    return pos, bold_edges


# ============================================================================
# METADATA
# ============================================================================

def get_planarity_metadata(G: nx.Graph, auxiliary, label: int) -> dict:
    """Generate metadata for planarity property."""
    is_planar, _ = nx.check_planarity(G)
    
    metadata = {
        'property': 'planarity',
        'label': label,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'ground_truth': {
            'is_planar': is_planar
        }
    }
    
    if label == 0 and auxiliary:  # Non-planar with minor or s_no template
        # Auxiliary can be:
        # - tuple/list: (minor_type, minor_nodes, minor_edges[, edge_paths]) for d_no
        # - dict: template positions mapping for s_no (node->(x,y))
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
        elif isinstance(auxiliary, dict):
            # s_no uses a template position dict to enforce YES layout; record its size
            metadata['ground_truth']['template_positions'] = len(auxiliary)

    return metadata
