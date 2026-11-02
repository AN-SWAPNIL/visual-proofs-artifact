"""
Cut-Vertex Property Generator

Generates graphs with/without cut-vertices following GRAPHTRAILS conventions.

Dataset structure:
- yes/: Graphs WITH cut-vertices (bold cut-vertex, dense spatial clusters)
- s_no/: Same layout as YES but 2-connected (no cut-vertex)  
- d_no/: Different layout (ear decomposition, flower-like pattern)

Generation strategy (v2):
- All graphs have 20-40 nodes
- NO graph is generated first, then s_no and d_no are derived from it
- YES graph is generated independently
"""

import numpy as np
import networkx as nx
import random
from typing import List, Tuple, Dict, Optional

# Global node range
MIN_NODES = 20
MAX_NODES = 40


def generate_cut_vertex_graph_yes(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a graph WITH cut-vertices using bridge strategy.
    
    Strategy: Create two Erdős-Rényi graphs connected by single bridge edge.
    The endpoints of the bridge become cut-vertices.
    
    Returns:
        G: NetworkX graph with cut-vertices
        cut_vertices: List of cut-vertex node indices
    """
    # Use global defaults if not specified
    min_n = min_nodes if min_nodes is not None else MIN_NODES
    max_n = max_nodes if max_nodes is not None else MAX_NODES
    
    max_attempts = 50
    
    for _ in range(max_attempts):
        # Total nodes in range, split into two components
        n = random.randint(min_n, max_n)
        n1 = random.randint(n // 3, 2 * n // 3)
        n2 = n - n1
        
        if n1 < 4 or n2 < 4:
            continue
        
        # Generate two Erdős-Rényi components with random edge probability
        p = random.uniform(0.25, 0.45)
        
        G1 = nx.erdos_renyi_graph(n1, p)
        G2 = nx.erdos_renyi_graph(n2, p)
        
        # Ensure both components are connected
        if not nx.is_connected(G1) or not nx.is_connected(G2):
            continue
        
        # Relabel G2 nodes to avoid overlap
        G2 = nx.relabel_nodes(G2, {i: i + n1 for i in G2.nodes()})
        
        # Combine graphs
        G = nx.compose(G1, G2)
        
        # Connect with a single bridge edge (creates two cut-vertices)
        bridge_node1 = random.choice(list(G1.nodes()))
        bridge_node2 = random.choice(list(G2.nodes()))
        G.add_edge(bridge_node1, bridge_node2)
        
        # Verify cut-vertices exist
        cut_verts = list(nx.articulation_points(G))
        
        if len(cut_verts) >= 1:
            # Store component info for layout
            G.graph['component1_nodes'] = list(G1.nodes())
            G.graph['component2_nodes'] = list(G2.nodes())
            G.graph['bridge_nodes'] = (bridge_node1, bridge_node2)
            return G, cut_verts
    
    # Fallback: simple bridge graph with more nodes
    n = random.randint(min_n, max_n)
    n1 = n // 2
    n2 = n - n1
    G = nx.Graph()
    # Component 1: cycle with chords
    G.add_edges_from([(i, (i + 1) % n1) for i in range(n1)])
    for _ in range(n1 // 3):
        u, v = random.sample(range(n1), 2)
        G.add_edge(u, v)
    # Component 2: cycle with chords
    G.add_edges_from([(n1 + i, n1 + (i + 1) % n2) for i in range(n2)])
    for _ in range(n2 // 3):
        u, v = random.sample(range(n1, n), 2)
        G.add_edge(u, v)
    # Bridge
    G.add_edge(n1 - 1, n1)
    G.graph['component1_nodes'] = list(range(n1))
    G.graph['component2_nodes'] = list(range(n1, n))
    G.graph['bridge_nodes'] = (n1 - 1, n1)
    return G, [n1 - 1, n1]


def generate_cut_vertex_graph_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict, List[List[int]]]:
    """
    Generate a 2-connected graph (NO cut-vertices) - base graph for both s_no and d_no.
    
    Strategy: Build using EXPLICIT ear decomposition:
    - Start with a base cycle P₀
    - Add ears P₁, P₂, ... as paths connecting two existing nodes
    - Track component-like structure for s_no layout
    
    Returns:
        G: NetworkX graph without cut-vertices (2-connected)
        info: Dict containing:
            - 'layout_info': Dict with pseudo-component info for s_no layout
            - 'ears': List of ears [P₀, P₁, P₂, ...] for d_no layout
    """
    # Use global defaults if not specified
    min_n = min_nodes if min_nodes is not None else MIN_NODES
    max_n = max_nodes if max_nodes is not None else MAX_NODES
    
    max_attempts = 100
    
    for _ in range(max_attempts):
        n = random.randint(min_n, max_n)
        
        # P₀: Start with a base cycle (8-12 nodes for larger graphs)
        cycle_size = random.randint(8, min(14, n // 2))
        G = nx.cycle_graph(cycle_size)
        
        # Track which nodes are on the "boundary" (can be ear endpoints)
        boundary_nodes = list(range(cycle_size))
        
        ears = [list(range(cycle_size))]  # P₀ = base cycle
        
        # Add remaining nodes as ears
        current_node = cycle_size
        nodes_remaining = n - cycle_size
        
        # Decide how many ears to create (3-7 ears for larger graphs)
        num_ears = random.randint(3, min(8, max(3, nodes_remaining // 2)))
        
        for ear_idx in range(num_ears):
            if current_node >= n:
                break
            
            # How many internal nodes in this ear (1-4 for variety)
            if ear_idx == num_ears - 1:
                ear_internal_count = n - current_node
            else:
                ear_internal_count = random.randint(1, min(4, n - current_node))
            
            if ear_internal_count < 1:
                continue
            
            # Choose two attachment points from boundary
            if len(boundary_nodes) >= 2:
                u_idx = random.randint(0, len(boundary_nodes) - 1)
                offset = random.randint(max(1, len(boundary_nodes) // 4), 
                                       max(2, len(boundary_nodes) // 2))
                v_idx = (u_idx + offset) % len(boundary_nodes)
                u = boundary_nodes[u_idx]
                v = boundary_nodes[v_idx]
            else:
                continue
            
            # Build ear: u -> internal nodes -> v
            ear_nodes = [u]
            
            for i in range(ear_internal_count):
                G.add_node(current_node)
                
                if i == 0:
                    G.add_edge(u, current_node)
                else:
                    G.add_edge(current_node - 1, current_node)
                
                ear_nodes.append(current_node)
                boundary_nodes.append(current_node)
                current_node += 1
            
            G.add_edge(current_node - 1, v)
            ear_nodes.append(v)
            ears.append(ear_nodes)
        
        # Verify NO cut-vertices
        cut_verts = list(nx.articulation_points(G))
        if len(cut_verts) == 0 and nx.is_connected(G) and G.number_of_nodes() >= min_n:
            # Create pseudo-component info for s_no layout
            # Split nodes roughly in half to create two "clusters"
            all_nodes = list(G.nodes())
            mid = len(all_nodes) // 2
            comp1_nodes = all_nodes[:mid]
            comp2_nodes = all_nodes[mid:]
            
            # Find a node that connects the two pseudo-components
            bridge_node = None
            for node in comp1_nodes:
                for neighbor in G.neighbors(node):
                    if neighbor in comp2_nodes:
                        bridge_node = node
                        break
                if bridge_node:
                    break
            if bridge_node is None:
                bridge_node = comp1_nodes[0]
            
            layout_info = {
                'component1_nodes': comp1_nodes,
                'component2_nodes': comp2_nodes,
                'bridge_position_node': bridge_node
            }
            
            return G, {'layout_info': layout_info, 'ears': ears}
    
    # Fallback: simple guaranteed ear decomposition with 20+ nodes
    n = min_n
    cycle_size = 10
    G = nx.cycle_graph(cycle_size)
    ears = [list(range(cycle_size))]
    current_node = cycle_size
    
    while current_node < n:
        ear_size = min(3, n - current_node)
        u = random.randint(0, current_node - 1)
        v = (u + cycle_size // 2) % min(current_node, cycle_size)
        ear_nodes = [u]
        for i in range(ear_size):
            G.add_node(current_node)
            if i == 0:
                G.add_edge(u, current_node)
            else:
                G.add_edge(current_node - 1, current_node)
            ear_nodes.append(current_node)
            current_node += 1
        G.add_edge(current_node - 1, v)
        ear_nodes.append(v)
        ears.append(ear_nodes)
    
    all_nodes = list(G.nodes())
    mid = len(all_nodes) // 2
    layout_info = {
        'component1_nodes': all_nodes[:mid],
        'component2_nodes': all_nodes[mid:],
        'bridge_position_node': all_nodes[mid - 1]
    }
    
    return G, {'layout_info': layout_info, 'ears': ears}


def generate_cut_vertex_graph_s_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, Dict]:
    """
    Generate a 2-connected graph (NO cut-vertices) for SAME LAYOUT as YES.
    
    Uses generate_cut_vertex_graph_no() and returns the graph with layout_info.
    The layout will look like two clusters but the graph is 2-connected.
    
    Returns:
        G: NetworkX graph without cut-vertices (2-connected)
        layout_info: Dict with component info for same-layout visualization
    """
    G, info = generate_cut_vertex_graph_no(min_nodes, max_nodes)
    return G, info['layout_info']


def generate_cut_vertex_graph_d_no(min_nodes: int = None, max_nodes: int = None) -> Tuple[nx.Graph, List[List[int]]]:
    """
    Generate a 2-connected graph (NO cut-vertices) for DIFFERENT LAYOUT (ear decomposition).
    
    Uses generate_cut_vertex_graph_no() and returns the graph with ears.
    The layout will show ear decomposition (flower pattern).
    
    Returns:
        G: NetworkX graph without cut-vertices (2-connected)
        ears: List of ears [P₀, P₁, P₂, ...] where P₀ is the base cycle
    """
    G, info = generate_cut_vertex_graph_no(min_nodes, max_nodes)
    return G, info['ears']


def layout_cut_vertex_yes(G: nx.Graph, cut_vertices: List[int]) -> Tuple[Dict[int, Tuple[float, float]], int]:
    """
    Layout for graphs WITH cut-vertices.
    
    Updated approach: 
    - Two DENSE spatial clusters (using spring layout, not circular)
    - Cut-vertex at bridge position between clusters
    - Returns position dict AND the cut-vertex node for bold styling
    
    Layout strategy:
    - Identify components separated by cut-vertex
    - Place component 1 at TOP-LEFT using spring layout (dense cluster)
    - Place component 2 at TOP-RIGHT using spring layout (dense cluster)
    - Place cut-vertex at BOTTOM-CENTER (bridge position)
    
    Returns:
        pos: Dict of node positions
        cut_vertex: The cut-vertex node (for bold styling)
    """
    if not cut_vertices:
        # Fallback to spring layout
        pos = nx.spring_layout(G, k=1.5, iterations=100, scale=2.5, seed=42)
        return pos, None
    
    # Use the first cut-vertex as the bridge point
    cut_v = cut_vertices[0]
    
    # Remove cut vertex and find components
    G_temp = G.copy()
    G_temp.remove_node(cut_v)
    components = list(nx.connected_components(G_temp))
    
    if len(components) < 2:
        # Fallback: use spring layout if removal doesn't create components
        pos = nx.spring_layout(G, k=1.5, iterations=100, scale=2.5, seed=42)
        return pos, cut_v
    
    pos = {}
    
    # Component 1: TOP-LEFT (spring layout for dense cluster appearance)
    comp1 = list(components[0])
    subgraph1 = G.subgraph(comp1).copy()
    # Use spring layout with high iterations for tight clustering
    pos1 = nx.spring_layout(subgraph1, k=0.8, iterations=100, scale=1.2, center=(-1.8, 1.5), seed=42)
    pos.update(pos1)
    
    # Component 2: TOP-RIGHT (spring layout for dense cluster appearance)
    comp2 = list(components[1])
    subgraph2 = G.subgraph(comp2).copy()
    pos2 = nx.spring_layout(subgraph2, k=0.8, iterations=100, scale=1.2, center=(1.8, 1.5), seed=43)
    pos.update(pos2)
    
    # Additional components if any: spread below
    for i, comp in enumerate(components[2:], start=2):
        comp_list = list(comp)
        subgraph = G.subgraph(comp_list).copy()
        x_offset = -2.0 + (i - 2) * 2.0
        pos_comp = nx.spring_layout(subgraph, k=0.8, iterations=100, scale=0.8, center=(x_offset, -1.5), seed=44+i)
        pos.update(pos_comp)
    
    # Cut-vertex: BOTTOM-CENTER (the bridge position)
    pos[cut_v] = (0.0, -0.5)
    
    return pos, cut_v


def layout_cut_vertex_s_no(G: nx.Graph, layout_info: Dict) -> Dict[int, Tuple[float, float]]:
    """
    Layout for 2-connected graphs using SAME LAYOUT as YES (for s_no class).
    
    Creates the visual appearance of two clusters connected by a bridge,
    but the graph is actually 2-connected (no cut-vertex).
    
    Layout strategy:
    - Place component 1 nodes at TOP-LEFT using spring layout
    - Place component 2 nodes at TOP-RIGHT using spring layout
    - Place the "bridge position node" at BOTTOM-CENTER
    
    Returns:
        pos: Dict of node positions
    """
    if not layout_info:
        # Fallback to spring layout
        return nx.spring_layout(G, k=1.5, iterations=100, scale=2.5, seed=42)
    
    comp1_nodes = layout_info.get('component1_nodes', [])
    comp2_nodes = layout_info.get('component2_nodes', [])
    bridge_node = layout_info.get('bridge_position_node', None)
    
    if not comp1_nodes or not comp2_nodes:
        return nx.spring_layout(G, k=1.5, iterations=100, scale=2.5, seed=42)
    
    pos = {}
    
    # Component 1: TOP-LEFT (excluding bridge node if present)
    comp1_without_bridge = [n for n in comp1_nodes if n != bridge_node]
    if comp1_without_bridge:
        subgraph1 = G.subgraph(comp1_without_bridge).copy()
        pos1 = nx.spring_layout(subgraph1, k=0.8, iterations=100, scale=1.2, center=(-1.8, 1.5), seed=42)
        pos.update(pos1)
    
    # Component 2: TOP-RIGHT (excluding bridge node if present)
    comp2_without_bridge = [n for n in comp2_nodes if n != bridge_node]
    if comp2_without_bridge:
        subgraph2 = G.subgraph(comp2_without_bridge).copy()
        pos2 = nx.spring_layout(subgraph2, k=0.8, iterations=100, scale=1.2, center=(1.8, 1.5), seed=43)
        pos.update(pos2)
    
    # Bridge position node: BOTTOM-CENTER (looks like cut-vertex but isn't)
    if bridge_node is not None:
        pos[bridge_node] = (0.0, -0.5)
    
    # Handle any remaining nodes
    for node in G.nodes():
        if node not in pos:
            # Place randomly in the middle area
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0.5, 1.0)
            pos[node] = (r * np.cos(angle), r * np.sin(angle))
    
    return pos


def layout_cut_vertex_d_no(G: nx.Graph, ears: List[List[int]]) -> Dict[int, Tuple[float, float]]:
    """
    Layout for 2-connected graphs using ear decomposition visualization.
    
    Following the reference image (flower-like layout):
    - P₀ (base cycle) forms a central circle
    - Each ear P₁, P₂, ... loops OUTWARD from the cycle at its attachment points
    - Each ear bulges away from the center, creating a flower petal pattern
    
    This clearly shows the biconnected structure.
    """
    if not ears or len(ears) == 0:
        # Fallback to circular layout
        return nx.circular_layout(G, scale=2.5)
    
    pos = {}
    
    # P₀: Base cycle - central circle
    base_cycle = ears[0]
    n_base = len(base_cycle)
    base_radius = 1.8  # Larger base cycle for better visibility
    
    # Place base cycle nodes evenly around a circle
    # Store angles for later use
    node_angles = {}
    for i, node in enumerate(base_cycle):
        angle = 2 * np.pi * i / n_base - np.pi / 2  # Start from top
        node_angles[node] = angle
        pos[node] = (base_radius * np.cos(angle), base_radius * np.sin(angle))
    
    # For each subsequent ear, place internal nodes on an outward arc
    for ear_idx, ear in enumerate(ears[1:], start=1):
        if len(ear) < 2:
            continue
        
        u, v = ear[0], ear[-1]  # Endpoints
        internal_nodes = ear[1:-1]
        
        # Skip if no internal nodes (chord)
        if not internal_nodes:
            continue
        
        # Get endpoint positions (they should already be placed)
        if u not in pos or v not in pos:
            # Fallback: place endpoints if missing
            angle = random.uniform(0, 2 * np.pi)
            if u not in pos:
                node_angles[u] = angle
                pos[u] = (base_radius * np.cos(angle), base_radius * np.sin(angle))
            if v not in pos:
                node_angles[v] = angle + np.pi/4
                pos[v] = (base_radius * np.cos(angle + np.pi/4), base_radius * np.sin(angle + np.pi/4))
        
        ux, uy = pos[u]
        vx, vy = pos[v]
        
        # Calculate the arc midpoint angle (average of endpoint angles)
        # This ensures the bulge goes OUTWARD from the cycle center
        angle_u = node_angles.get(u, np.arctan2(uy, ux))
        angle_v = node_angles.get(v, np.arctan2(vy, vx))
        
        # Handle angle wrapping - find the shorter arc
        angle_diff = angle_v - angle_u
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Midpoint angle on the arc
        mid_angle = angle_u + angle_diff / 2
        
        # Outward direction from center at this angle
        out_dx = np.cos(mid_angle)
        out_dy = np.sin(mid_angle)
        
        # Chord length affects bulge - longer chords need more bulge
        chord_len = np.sqrt((vx - ux)**2 + (vy - uy)**2)
        
        # Minimum bulge to avoid straight lines
        min_bulge = 0.8
        
        # Bulge distance varies by ear index for visual separation
        # Also scale with chord length to keep proportional
        bulge_base = max(min_bulge, 0.8 + 0.4 * chord_len + 0.3 * (ear_idx % 3))
        
        # Place internal nodes along a curved arc bulging outward
        n_internal = len(internal_nodes)
        
        for i, node in enumerate(internal_nodes):
            # Parameter along the ear (0 to 1)
            t = (i + 1) / (n_internal + 1)
            
            # Linear interpolation between endpoints
            base_x = ux + t * (vx - ux)
            base_y = uy + t * (vy - uy)
            
            # Parabolic bulge (maximum at middle of ear)
            # Ensure minimum bulge even at endpoints of internal nodes
            bulge_factor = 4 * t * (1 - t)  # Max = 1.0 at t=0.5
            bulge_factor = max(bulge_factor, 0.3)  # Minimum bulge factor
            
            # Position = base + outward bulge
            node_x = base_x + out_dx * bulge_base * bulge_factor
            node_y = base_y + out_dy * bulge_base * bulge_factor
            
            pos[node] = (node_x, node_y)
            
            # Store angle for this node (for potential nested ears)
            node_angles[node] = np.arctan2(node_y, node_x)
    
    # Handle any remaining nodes not in ears (shouldn't happen with proper generation)
    for node in G.nodes():
        if node not in pos:
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(base_radius + 0.5, base_radius + 1.5)
            pos[node] = (r * np.cos(angle), r * np.sin(angle))
    
    return pos


def get_cut_vertex_metadata(G: nx.Graph, auxiliary_data, label: int) -> dict:
    """
    Generate metadata for cut-vertex property.
    
    Args:
        auxiliary_data: For YES class, this is cut_vertices list.
                       For s_no class, this is layout_info dict.
                       For d_no class, this is ears list (ear decomposition).
        label: 1 for YES, 0 for NO (s_no or d_no)
    """
    metadata = {
        'property': 'cut_vertex',
        'label': label,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'ground_truth': {}
    }
    
    if label == 1:  # YES - has cut vertices
        cut_vertices = auxiliary_data if isinstance(auxiliary_data, list) else []
        metadata['ground_truth']['cut_vertices'] = cut_vertices
        metadata['ground_truth']['num_cut_vertices'] = len(cut_vertices)
    else:  # NO - 2-connected
        metadata['ground_truth']['is_2_connected'] = True
        if isinstance(auxiliary_data, dict):
            # s_no class - layout_info dict
            metadata['ground_truth']['layout_type'] = 'same_as_yes'
        elif isinstance(auxiliary_data, list):
            # d_no class - ears list
            metadata['ground_truth']['layout_type'] = 'ear_decomposition'
            metadata['ground_truth']['num_ears'] = len(auxiliary_data)
    
    return metadata


# Backward compatibility aliases - use the real d_no generator directly
def layout_cut_vertex_no(G: nx.Graph, ears: List[List[int]]) -> Dict[int, Tuple[float, float]]:
    """Backward compatibility: calls layout_cut_vertex_d_no."""
    return layout_cut_vertex_d_no(G, ears)
