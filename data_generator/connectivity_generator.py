"""
Connectivity Property Generator

Generates connected/disconnected graphs following GRAPHTRAILS Fig 3c conventions:
- BFS spanning tree with concentric circles layout
- Vertices on circles based on distance from root
- Bold spanning tree edges (same style for all components)
- Visual proof: YES = single centered tree, NO = multiple trees

3-class structure:
- YES: Connected graph with bold spanning tree
- s_no: Disconnected graph with SAME layout as YES (all components centered), only bold largest component
- d_no: Disconnected graph with DIFFERENT layout (components spatially separated), no bold edges

Generation strategy (v3 - Enhanced Randomization):
- Graphs have 15-60 nodes (wider range for unpredictability)
- Multiple graph generation algorithms for variety
- Per-image unique seeding for randomness
- NO graph is generated first, then s_no and d_no are derived from it
- YES graph is generated independently
"""

import numpy as np
import networkx as nx
import random
from typing import List, Tuple, Dict, Set

# Import randomization config (with fallback)
try:
    from randomization_config import (
        get_random_node_range, sample_edge_probability,
        sample_visual_params, CONFIG
    )
    USE_RANDOMIZATION = True
except ImportError:
    USE_RANDOMIZATION = False

# Global node range (wider for variety)
MIN_NODES = 15  # Reduced from 20
MAX_NODES = 60  # Increased from 40


# ============================================================================
# HELPER FUNCTIONS (DRY - Don't Repeat Yourself)
# ============================================================================

def _random_partition(total: int, num_parts: int, min_size: int = 4) -> List[int]:
    """Randomly partition total into num_parts, each >= min_size."""
    if total < num_parts * min_size:
        return None
    
    sizes = []
    remaining = total
    
    for i in range(num_parts - 1):
        max_for_this = remaining - min_size * (num_parts - i - 1)
        if max_for_this < min_size:
            return None
        size = random.randint(min_size, max_for_this)
        sizes.append(size)
        remaining -= size
    
    sizes.append(remaining)
    random.shuffle(sizes)
    return sizes


def _generate_random_connected_component(size: int) -> nx.Graph:
    """Generate a random connected graph of given size with variety."""
    max_attempts = 20
    
    for _ in range(max_attempts):
        graph_type = random.choice(['erdos_renyi', 'watts_strogatz', 'barabasi_albert'])
        
        try:
            if graph_type == 'erdos_renyi':
                p = random.uniform(0.3, 0.55)
                G = nx.erdos_renyi_graph(size, p)
                
            elif graph_type == 'watts_strogatz':
                k = min(random.choice([3, 4]), size - 1)
                p = random.uniform(0.2, 0.5)
                G = nx.connected_watts_strogatz_graph(size, k, p)
                
            else:  # barabasi_albert
                m = min(random.randint(2, 3), size - 1)
                G = nx.barabasi_albert_graph(size, m)
            
            if nx.is_connected(G):
                return G
                
        except Exception:
            continue
    
    # Fallback: cycle graph
    return nx.cycle_graph(size)


def _compute_all_spanning_trees(G: nx.Graph) -> List[Tuple[int, int]]:
    """Compute BFS spanning tree for each connected component."""
    spanning_tree_edges = []
    
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component).copy()
        root = random.choice(list(component))
        bfs_tree = nx.bfs_tree(subgraph, source=root)
        spanning_tree_edges.extend(list(bfs_tree.edges()))
    
    return spanning_tree_edges


def _layout_component_concentric(
    G: nx.Graph,
    component: Set[int],
    center_x: float = 0.0,
    center_y: float = 0.0
) -> Dict[int, Tuple[float, float]]:
    """
    Layout a single component using concentric circles (GRAPHTRAILS Fig 3c).
    UNIFIED function used by both YES and NO layouts.
    """
    subgraph = G.subgraph(component).copy()
    
    # Choose random root
    root = random.choice(list(component))
    
    # Compute BFS levels
    levels = {root: 0}
    queue = [root]
    visited = {root}
    
    while queue:
        node = queue.pop(0)
        for neighbor in subgraph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                levels[neighbor] = levels[node] + 1
                queue.append(neighbor)
    
    # Group nodes by level
    nodes_by_level = {}
    for node, level in levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    max_level = max(levels.values()) if levels else 0
    pos = {}
    
    # Randomize starting angle for variety
    angle_offset = random.uniform(0, 2 * np.pi)
    
    # Place nodes on concentric circles
    for level in range(max_level + 1):
        nodes_at_level = nodes_by_level.get(level, [])
        num_nodes = len(nodes_at_level)
        
        if num_nodes == 0:
            continue
        
        # Radius increases with level
        if level == 0:
            radius = 0.2  # Root at center
        else:
            radius = 0.8 + 0.7 * level
        
        # Place nodes evenly around circle
        for i, node in enumerate(nodes_at_level):
            if num_nodes == 1:
                angle = angle_offset
            else:
                angle = angle_offset + 2 * np.pi * i / num_nodes
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            pos[node] = (x, y)
    
    return pos


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_connectivity_graph_yes(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List]:
    """
    Generate a CONNECTED graph with BFS spanning tree.
    Uses variety of graph types for randomization.
    
    Returns:
        G: Connected NetworkX graph
        spanning_tree_edges: List of edges in the BFS spanning tree
    """
    # Use global defaults if not specified
    min_n = min_nodes if min_nodes is not None else MIN_NODES
    max_n = max_nodes if max_nodes is not None else MAX_NODES
    
    max_attempts = 50
    
    for _ in range(max_attempts):
        n = random.randint(min_n, max_n)
        
        # Randomize graph type for variety
        graph_type = random.choice(['watts_strogatz', 'barabasi_albert', 'erdos_renyi', 'random_regular'])
        
        try:
            if graph_type == 'watts_strogatz':
                k = random.choice([4, 6])
                if k >= n:
                    k = max(2, n - 1)
                p = random.uniform(0.15, 0.45)
                G = nx.connected_watts_strogatz_graph(n, k, p)
                
            elif graph_type == 'barabasi_albert':
                m = random.randint(2, 4)
                G = nx.barabasi_albert_graph(n, m)
                
            elif graph_type == 'erdos_renyi':
                p = random.uniform(0.25, 0.45)
                G = nx.erdos_renyi_graph(n, p)
                if not nx.is_connected(G):
                    continue
                    
            else:  # random_regular
                d = random.choice([3, 4, 5])
                if d >= n or (n * d) % 2 != 0:
                    d = 4 if n % 2 == 0 else 3
                G = nx.random_regular_graph(d, n)
            
            if nx.is_connected(G):
                root = random.choice(list(G.nodes()))
                bfs_tree = nx.bfs_tree(G, source=root)
                return G, list(bfs_tree.edges())
                
        except Exception:
            continue
    
    # Fallback: cycle graph with extra edges
    n = random.randint(min_n, max_n)
    G = nx.cycle_graph(n)
    for _ in range(n // 2):
        u, v = random.sample(list(G.nodes()), 2)
        G.add_edge(u, v)
    
    root = random.choice(list(G.nodes()))
    bfs_tree = nx.bfs_tree(G, source=root)
    return G, list(bfs_tree.edges())


def generate_connectivity_graph_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a DISCONNECTED graph (2+ components) - base graph for both s_no and d_no.
    
    Returns:
        G: Disconnected NetworkX graph
        info: Dict with 'largest_component_tree_edges' for s_no bolding
              and 'all_components' for layout
    """
    # Use global defaults if not specified
    min_n = min_nodes if min_nodes is not None else MIN_NODES
    max_n = max_nodes if max_nodes is not None else MAX_NODES
    
    max_attempts = 50
    
    for _ in range(max_attempts):
        num_components = random.choice([2, 2, 2, 3])  # Bias toward 2
        total_nodes = random.randint(min_n, max_n)
        
        component_sizes = _random_partition(total_nodes, num_components, min_size=6)
        if component_sizes is None:
            continue
        
        # Generate components with variety
        G = nx.Graph()
        node_offset = 0
        
        for size in component_sizes:
            comp = _generate_random_connected_component(size)
            if comp is None:
                continue
            
            comp = nx.relabel_nodes(comp, {i: i + node_offset for i in comp.nodes()})
            G = nx.compose(G, comp)
            node_offset += size
        
        if nx.number_connected_components(G) >= 2:
            # Find largest component
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            subgraph = G.subgraph(largest_component).copy()
            
            # Get spanning tree of largest component only (for bolding in s_no)
            root = random.choice(list(largest_component))
            bfs_tree = nx.bfs_tree(subgraph, source=root)
            
            info = {
                'largest_component': largest_component,
                'largest_component_tree_edges': list(bfs_tree.edges()),
                'all_components': components
            }
            
            return G, info
    
    # Fallback: two cycles
    n = random.randint(min_n, max_n)
    n1 = n // 2
    n2 = n - n1
    G1 = nx.cycle_graph(n1)
    G2 = nx.cycle_graph(n2)
    G2 = nx.relabel_nodes(G2, {i: i + n1 for i in G2.nodes()})
    G = nx.compose(G1, G2)
    
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    subgraph = G.subgraph(largest_component).copy()
    root = random.choice(list(largest_component))
    bfs_tree = nx.bfs_tree(subgraph, source=root)
    
    info = {
        'largest_component': largest_component,
        'largest_component_tree_edges': list(bfs_tree.edges()),
        'all_components': components
    }
    
    return G, info


def generate_connectivity_graph_s_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a DISCONNECTED graph for SAME layout as YES.
    Uses generate_connectivity_graph_no() and returns info for s_no layout.
    
    Returns:
        G: Disconnected NetworkX graph
        layout_info: Dict with 'largest_component_tree_edges' for bolding
    """
    G, info = generate_connectivity_graph_no(min_nodes, max_nodes)
    return G, info


def generate_connectivity_graph_d_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, None]:
    """
    Generate a DISCONNECTED graph for DIFFERENT layout (spatially separated).
    Uses generate_connectivity_graph_no() - no bold edges for d_no.
    
    Returns:
        G: Disconnected NetworkX graph
        None: No auxiliary data (no bolding)
    """
    G, _ = generate_connectivity_graph_no(min_nodes, max_nodes)
    return G, None


# ============================================================================
# LAYOUT FUNCTIONS (using unified helper)
# ============================================================================

def layout_connectivity_yes(G: nx.Graph, spanning_tree_edges: List) -> Tuple[Dict[int, Tuple[float, float]], Set]:
    """
    GRAPHTRAILS Fig 3c layout for CONNECTED graphs.
    
    Paper approach: Spanning tree is rooted at central vertex with vertices
    drawn on concentric circles depending on distance from root.
    The spanning tree is crossing-free in this layout.
    
    Visual proof: Single spanning tree connects ALL vertices, shown clearly
    with concentric circles emanating from a central root.
    
    CRITICAL: Uses angular sector assignment to GUARANTEE no spanning tree edge crossings.
    Each node gets an angular sector, and all its descendants are placed within that sector.
    
    Returns:
        Tuple of (pos, bold_edges) where bold_edges is the spanning tree edges to bold
    """
    if not spanning_tree_edges:
        pos = _layout_component_concentric(G, set(G.nodes()), center_x=0.0, center_y=0.0)
        return pos, set()
    
    # Build directed tree from spanning tree edges
    tree = nx.Graph()
    tree.add_edges_from(spanning_tree_edges)
    
    # Find root (pick a node - use one with many neighbors for balanced tree)
    degree_dict = dict(tree.degree())
    root = max(tree.nodes(), key=lambda n: degree_dict[n])
    
    # Build rooted tree structure: for each node, find its children
    parent = {root: None}
    children = {node: [] for node in tree.nodes()}
    queue = [root]
    visited = {root}
    levels = {root: 0}
    
    while queue:
        node = queue.pop(0)
        for neighbor in tree.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                levels[neighbor] = levels[node] + 1
                queue.append(neighbor)
    
    # Add any nodes not in spanning tree (isolated)
    for node in G.nodes():
        if node not in visited:
            levels[node] = 1
            visited.add(node)
    
    max_level = max(levels.values()) if levels else 0
    
    # Count descendants for each node (used for angular allocation)
    def count_descendants(node):
        count = 1  # count self
        for child in children[node]:
            count += count_descendants(child)
        return count
    
    descendant_count = {node: count_descendants(node) for node in tree.nodes()}
    
    # Place nodes using angular sector allocation to prevent crossings
    pos = {}
    angle_offset = random.uniform(0, 2 * np.pi)
    
    def place_subtree(node, angle_start, angle_end, level):
        """Recursively place a subtree within an angular sector."""
        # Radius increases with level
        if level == 0:
            radius = 0.0  # Root at center
        else:
            radius = 1.0 + 0.9 * level  # Good spacing between levels
        
        # Place this node at the center of its angular sector
        angle = (angle_start + angle_end) / 2
        x = radius * np.cos(angle + angle_offset)
        y = radius * np.sin(angle + angle_offset)
        pos[node] = (x, y)
        
        # Divide angular sector among children proportionally to their descendant counts
        if children[node]:
            total_descendants = sum(descendant_count[c] for c in children[node])
            current_angle = angle_start
            
            for child in children[node]:
                # Allocate angular sector proportional to number of descendants
                child_ratio = descendant_count[child] / total_descendants
                child_angle_span = (angle_end - angle_start) * child_ratio
                child_angle_end = current_angle + child_angle_span
                
                place_subtree(child, current_angle, child_angle_end, level + 1)
                current_angle = child_angle_end
    
    # Place the entire tree
    place_subtree(root, 0, 2 * np.pi, 0)
    
    # Return both pos and bold_edges (spanning tree edges)
    bold_edges = set((u, v) for u, v in spanning_tree_edges)
    return pos, bold_edges


def layout_connectivity_s_no(G: nx.Graph, layout_info: Dict) -> Tuple[Dict[int, Tuple[float, float]], Set]:
    """
    Layout for s_no class: SAME layout as YES (concentric circles).
    
    ALL nodes laid out TOGETHER in one unified concentric circle layout.
    Components are MIXED (not separated) - same visual style as YES.
    Bold edges for the LARGEST component's spanning tree only.
    
    Visual test: Model must detect that bold edges don't connect ALL nodes.
    
    Returns:
        Tuple of (pos, bold_edges) where bold_edges is largest component's spanning tree
    """
    # Choose a random root from the largest component
    all_components = layout_info.get('all_components', list(nx.connected_components(G)))
    largest_component = max(all_components, key=len)
    root = random.choice(list(largest_component))
    
    # Perform BFS from root on the ACTUAL graph (which is disconnected)
    # This will reach only the root's component, leaving others unreached
    levels = {root: 0}
    queue = [root]
    visited = {root}
    children = {node: [] for node in G.nodes()}
    
    while queue:
        node = queue.pop(0)
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                levels[neighbor] = levels[node] + 1
                children[node].append(neighbor)
                queue.append(neighbor)
    
    # For unreached nodes (other components), assign them to random levels
    # and distribute them around the circle to MIX with the main component
    unreached_nodes = set(G.nodes()) - visited
    if unreached_nodes:
        max_level = max(levels.values()) if levels else 0
        for node in unreached_nodes:
            # Assign random level between 1 and max_level+1
            levels[node] = random.randint(1, max_level + 2)
    
    # Count descendants for angular allocation
    def count_descendants(node):
        count = 1
        for child in children[node]:
            count += count_descendants(child)
        return count
    
    descendant_count = {node: count_descendants(node) for node in G.nodes()}
    
    # Place nodes using angular sector allocation (SAME as YES)
    pos = {}
    angle_offset = random.uniform(0, 2 * np.pi)
    
    def place_subtree(node, angle_start, angle_end, level):
        if level == 0:
            radius = 0.0
        else:
            radius = 1.0 + 0.9 * level
        
        angle = (angle_start + angle_end) / 2
        x = radius * np.cos(angle + angle_offset)
        y = radius * np.sin(angle + angle_offset)
        pos[node] = (x, y)
        
        if children[node]:
            total_descendants = sum(descendant_count[c] for c in children[node])
            current_angle = angle_start
            
            for child in children[node]:
                child_ratio = descendant_count[child] / total_descendants
                child_angle_span = (angle_end - angle_start) * child_ratio
                child_angle_end = current_angle + child_angle_span
                place_subtree(child, current_angle, child_angle_end, level + 1)
                current_angle = child_angle_end
    
    # Place the main component's tree
    place_subtree(root, 0, 2 * np.pi, 0)
    
    # Place unreached nodes at their assigned levels with random angles
    for node in unreached_nodes:
        level = levels[node]
        radius = 1.0 + 0.9 * level if level > 0 else 0.0
        angle = random.uniform(0, 2 * np.pi) + angle_offset
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[node] = (x, y)
    
    # Get bold edges from largest component ONLY
    bold_edges = set()
    if layout_info and 'largest_component_tree_edges' in layout_info:
        bold_edges = set((u, v) for u, v in layout_info['largest_component_tree_edges'])
    
    return pos, bold_edges


def layout_connectivity_d_no(G: nx.Graph, auxiliary: None = None) -> Dict[int, Tuple[float, float]]:
    """
    Layout for d_no class: Simple spatial separation (NOT concentric circles).
    
    Components are laid out as TIGHT CLUSTERS with LARGE GAPS between them.
    - Component A: LEFT side (tight cluster)
    - Component B: RIGHT side (tight cluster)
    - LARGE GAP in middle (clearly visible isolation)
    
    NO bold edges - visual proof is the spatial separation itself.
    
    Why NOT concentric circles:
    - Concentric circles = "root can reach all nodes" = looks CONNECTED
    - For NO case, we need visual separation showing ISOLATED clusters
    
    Returns:
        pos: Node positions dictionary (no bold edges)
    """
    components = list(nx.connected_components(G))
    num_components = len(components)
    
    if num_components == 1:
        # Fallback: single component, use spring layout
        return nx.spring_layout(G, k=0.5, iterations=50, seed=random.randint(0, 10000))
    
    pos = {}
    
    # Sort components by size (largest first)
    components = sorted(components, key=len, reverse=True)
    
    # Define cluster centers with LARGE gaps between them
    if num_components == 2:
        # Two components: LEFT and RIGHT with large gap
        centers = [(-3.0, 0.0), (3.0, 0.0)]
    elif num_components == 3:
        # Three components: LEFT, TOP-RIGHT, BOTTOM-RIGHT
        centers = [(-3.0, 0.0), (2.5, 2.0), (2.5, -2.0)]
    else:
        # Multiple components: spread them out
        centers = []
        angle_step = 2 * np.pi / num_components
        radius = 3.0
        for i in range(num_components):
            angle = i * angle_step + random.uniform(-0.2, 0.2)
            centers.append((radius * np.cos(angle), radius * np.sin(angle)))
    
    # Layout each component as a TIGHT CLUSTER using spring layout
    for i, component in enumerate(components):
        comp_nodes = list(component)
        subgraph = G.subgraph(comp_nodes)
        
        # Use spring layout for cluster
        comp_pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=random.randint(0, 10000))
        
        # Scale to make cluster visible but separated, and shift to center position
        cx, cy = centers[i] if i < len(centers) else (random.uniform(-3, 3), random.uniform(-3, 3))
        scale = 1.5  # Larger scale so edges are visible
        
        for node in comp_nodes:
            x, y = comp_pos[node]
            pos[node] = (x * scale + cx, y * scale + cy)
    
    return pos


# Backward compatibility wrapper
def layout_connectivity_no(G: nx.Graph, spanning_tree_edges: List = None) -> Dict[int, Tuple[float, float]]:
    """Backward compatibility wrapper - uses d_no layout."""
    return layout_connectivity_d_no(G, None)


# ============================================================================
# METADATA
# ============================================================================

def get_connectivity_metadata(G: nx.Graph, auxiliary_data, label: int) -> dict:
    """Generate metadata for connectivity property.
    
    Args:
        G: The graph
        auxiliary_data: Can be spanning_tree_edges (list) or layout_info (dict) or None
        label: 1 for connected, 0 for disconnected
    """
    metadata = {
        'property': 'connectivity',
        'label': label,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'ground_truth': {
            'is_connected': label == 1,
            'num_components': nx.number_connected_components(G)
        }
    }
    
    # Handle different auxiliary_data types
    if auxiliary_data is not None:
        if isinstance(auxiliary_data, list):
            # spanning_tree_edges
            metadata['ground_truth']['spanning_tree_size'] = len(auxiliary_data)
        elif isinstance(auxiliary_data, dict) and 'largest_component_tree_edges' in auxiliary_data:
            # layout_info for s_no
            metadata['ground_truth']['spanning_tree_size'] = len(auxiliary_data['largest_component_tree_edges'])
    
    return metadata
