"""
MLLM Reasoning Text Generator

Generates detailed reasoning explanations for graph property classification.
These texts explain WHY a graph has or doesn't have a property based on 
visual certificates, enabling better MLLM fine-tuning.

Each property has templates for:
- YES responses with visual proof explanation
- s_no responses (same layout, negative) with reasoning
- d_no responses (different layout, negative) with reasoning

Usage:
    from reasoning_text_generator import generate_reasoning_text
    
    text = generate_reasoning_text(
        property_name='cut_vertex',
        label='yes',  # or 's_no' or 'd_no'
        G=graph,
        auxiliary_data=aux_data
    )
"""

import random
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any

# ============================================================================
# REASONING TEXT TEMPLATES
# ============================================================================

# Template variables use {var_name} format for substitution
REASONING_TEMPLATES = {
    'cut_vertex': {
        'yes': [
            "This graph contains a cut vertex (articulation point). The BOLD highlighted node is a cut vertex - removing it would disconnect the graph into {num_components} separate components. The two dense clusters are connected only through this critical node.",
            "The graph has a cut vertex shown in BOLD. This node acts as a bridge between {num_components} components. Without this node, the graph would split into disconnected parts with {comp1_size} and {comp2_size} nodes respectively.",
            "A cut vertex exists at the BOLD node position. The graph structure shows two clusters that depend entirely on this single node for connectivity. Removing it creates {num_components} isolated components.",
            "The BOLD node is an articulation point. The visual shows it connecting separate dense regions. The graph would fragment into {num_components} parts if this node were removed.",
        ],
        's_no': [
            "This graph has no cut vertex. Although the layout resembles a bridge structure with two clusters, the graph is 2-connected. Multiple paths exist between any pair of nodes, so removing any single node keeps the graph connected.",
            "No cut vertex exists. The visual layout shows two cluster-like regions, but they are interconnected by multiple edges. The graph remains connected after removing any single node.",
            "The graph is 2-connected with no articulation points. Despite the visual clustering, redundant edges ensure every node has at least 2 independent paths to any other node.",
            "No single node removal can disconnect this graph. The bridge-like appearance is misleading - internal edges within and between clusters provide backup connectivity.",
        ],
        'd_no': [
            "This graph is 2-connected, proven by the ear decomposition (flower pattern). The structure shows {num_ears} ears spreading from the central cycle. Every node lies on at least 2 independent paths to any other node.",
            "The ear decomposition visualization proves 2-connectivity. The base cycle and {num_ears} attached ears form a redundant structure with no cut vertices.",
            "No cut vertex exists. The flower-like pattern shows ears bulging outward from a central cycle. This ear decomposition is a certificate of 2-connectivity.",
            "The graph's ear decomposition proves no articulation point exists. Each ear adds an independent path, making the structure robust against single node removal.",
        ],
    },
    
    'connectivity': {
        'yes': [
            "The graph is connected. The BFS spanning tree (BOLD edges) successfully reaches all {num_nodes} nodes from the root. Every node is accessible from every other node.",
            "All {num_nodes} nodes form a single connected component. The concentric circle layout shows the BFS tree spanning the entire graph with no isolated vertices.",
            "Connected graph with {num_nodes} nodes and {num_edges} edges. The BOLD spanning tree edges demonstrate a path exists between any pair of nodes.",
            "The graph is fully connected. Starting from the center, the BFS tree reaches every node as shown by the BOLD edges forming a spanning structure.",
        ],
        's_no': [
            "The graph is disconnected with {num_components} separate components. The largest component (with BOLD spanning tree) contains {largest_size} nodes, but {isolated_nodes} nodes are unreachable.",
            "Multiple disconnected components exist. The BFS spanning tree (BOLD) only covers the main component of {largest_size} nodes. Other components totaling {other_nodes} nodes are isolated.",
            "The graph has {num_components} components. Although the layout is similar to a connected graph, the spanning tree cannot reach all nodes - gaps indicate missing connections.",
            "Disconnected graph with {num_components} separate parts. The BOLD edges show the spanning tree of one component, leaving {isolated_nodes} nodes in other components unreached.",
        ],
        'd_no': [
            "The graph is clearly disconnected. The visualization shows {num_components} spatially separated clusters with visible gaps between them. No edges connect these isolated regions.",
            "{num_components} disconnected components are clearly visible as separate spatial clusters. The gaps between clusters prove no path exists between them.",
            "The spatially separated layout reveals {num_components} isolated components with sizes {component_sizes}. Each cluster is internally connected but has no edges to other clusters.",
            "Disconnected graph evident from the spatial separation. {num_components} distinct clusters with clear gaps between them. Total nodes: {num_nodes}, distributed across isolated components.",
        ],
    },
    
    'bipartite': {
        'yes': [
            "The graph is bipartite (2-colorable). The adjacency matrix shows exactly 2 empty diagonal blocks - sets U ({set_u_size} nodes) and V ({set_v_size} nodes) are independent sets with no internal edges.",
            "Bipartite structure confirmed. The matrix has 2 empty diagonal blocks representing the two partitions. All edges exist only between partitions (colored cells in off-diagonal blocks).",
            "The graph can be 2-colored. The {num_nodes}-node graph partitions into sets of {set_u_size} and {set_v_size} nodes. Diagonal blocks are empty, proving no odd cycles exist.",
            "Bipartite graph verified by matrix structure. Two independent sets form empty diagonal blocks. Every edge connects nodes from different partitions.",
        ],
        's_no': [
            "The graph is NOT bipartite. Although sorted into 2 blocks, the diagonal blocks contain colored cells indicating edges within partitions. This violates the bipartite property.",
            "Not bipartite despite the 2-block structure. Intra-partition edges (visible as colored cells in diagonal blocks) prove nodes cannot be 2-colored without adjacent same-color nodes.",
            "The 2-block layout is misleading - this is non-bipartite. Edges exist within what would be color classes (diagonal block entries), making 2-coloring impossible.",
            "Non-bipartite graph. The matrix shows edges inside the diagonal blocks (colored cells), indicating odd cycles exist. A valid 2-partition is impossible.",
        ],
        'd_no': [
            "Contains an odd cycle of length {cycle_length}. The highlighted cells in the top-left block trace the cycle: {cycle_description}. Odd cycles prevent bipartiteness.",
            "Non-bipartite due to odd cycle. The {cycle_length}-node cycle is highlighted in the matrix. Nodes form a closed path of odd length, making 2-coloring impossible.",
            "The highlighted odd cycle proves non-bipartiteness. The cycle of length {cycle_length} appears as a connected sequence in the matrix's top-left region.",
            "An odd-length cycle ({cycle_length} nodes) is highlighted. This cycle cannot be 2-colored since alternating colors return to the start with a conflict.",
        ],
    },
    
    'planarity': {
        'yes': [
            "The graph is planar. The embedding shows all {num_edges} edges drawn without any crossings. The layout achieves a valid planar representation.",
            "Planar graph with {num_nodes} nodes and {num_edges} edges. The crossing-free embedding proves planarity - no two edges intersect except at shared endpoints.",
            "Valid planar embedding achieved. The spring/planar layout positions all nodes such that edges do not cross. This proves the graph can be drawn on a plane.",
            "Planar structure verified. The clean layout with {num_edges} non-crossing edges demonstrates this graph contains no K₅ or K₃,₃ minor.",
        ],
        's_no': [
            "The graph is non-planar. Although using a planar-style layout, some edges visibly CROSS each other. A planar embedding is impossible for this graph.",
            "Non-planar despite the spread layout. Edge crossings are visible where lines intersect away from vertices. The graph contains a Kuratowski substructure.",
            "The crossing edges prove non-planarity. The same-layout visualization cannot avoid intersections because the graph structure inherently requires crossings.",
            "Not planar - edge crossings are unavoidable. The planar-style layout reveals intrinsic non-planarity through visible intersecting edges.",
        ],
        'd_no': [
            "Contains a {minor_type} minor (BOLD edges). These {minor_size} nodes form a complete subgraph that cannot be embedded without crossings, proving non-planarity.",
            "Non-planar proven by {minor_type} minor. The highlighted structure shows the forbidden minor with all required edges present (BOLD).",
            "The {minor_type} minor (in BOLD) is the certificate of non-planarity. By Kuratowski's theorem, graphs containing K₅ or K₃,₃ are non-planar.",
            "{minor_type} subdivision found and highlighted. The BOLD edges connect the {minor_size} minor vertices, proving a planar embedding is impossible.",
        ],
    },
    
    'outer_planarity': {
        'yes': [
            "The graph is outer-planar. All {num_nodes} vertices lie on the outer face (circle boundary) with zero edge crossings. The graph can be drawn with all vertices on a single face.",
            "Outer-planar structure achieved. Every node is placed on the outer boundary, and all {num_edges} edges are drawn inside without crossings.",
            "Valid outer-planar embedding. The circular layout shows all vertices on the outer face. No K₄ or K₂,₃ minor exists in this graph.",
            "The graph is outer-planar with {num_nodes} outer-face vertices. The edge arrangement inside the circle has no crossings, proving outer-planarity.",
        ],
        's_no': [
            "Not outer-planar. Despite the circular layout, some edges visibly CROSS inside the circle. An outer-planar embedding is impossible.",
            "Non-outer-planar - edge crossings occur. The graph cannot place all vertices on the outer face while keeping edges non-crossing.",
            "The crossing edges inside the circle prove non-outer-planarity. Although vertices are on the boundary, the edge structure requires internal crossings.",
            "Not outer-planar. The same circular layout reveals unavoidable edge crossings, indicating a K₄ or K₂,₃ minor exists.",
        ],
        'd_no': [
            "Contains a {minor_type} minor (BOLD edges). This {minor_size}-vertex structure proves non-outer-planarity since {minor_type} cannot be outer-planar.",
            "Non-outer-planar due to {minor_type} minor. The highlighted edges show the forbidden substructure that prevents outer-planar embedding.",
            "The {minor_type} minor is highlighted with BOLD edges. By the forbidden minor characterization, this graph is not outer-planar.",
            "{minor_type} subdivision found. The BOLD edges mark this forbidden structure, proving outer-planar embedding is impossible.",
        ],
    },
    
    'hamiltonian': {
        'yes': [
            "The graph has a Hamiltonian cycle. The BOLD edges form a cycle visiting all {num_nodes} nodes exactly once, then returning to the start.",
            "Hamiltonian cycle exists. All {num_nodes} nodes lie on the outer circle because they all belong to the Hamiltonian path shown by BOLD edges.",
            "A valid Hamiltonian cycle (BOLD) visits every vertex exactly once. The cycle spans all {num_nodes} nodes before returning to its origin.",
            "Hamiltonian proved by the BOLD cycle. Every node appears exactly once on this cycle of length {num_nodes}, forming a complete tour.",
        ],
        's_no': [
            "No Hamiltonian cycle exists. The longest cycle found (BOLD edges) covers only {cycle_size} of {num_nodes} nodes. The remaining {interior_nodes} nodes are inside the cycle.",
            "Not Hamiltonian. The outer circle shows the longest cycle of {cycle_size} nodes, but {interior_nodes} nodes remain in the interior, unreachable by any extension.",
            "The graph lacks a Hamiltonian cycle. The longest cycle (BOLD) has {cycle_size} nodes; {interior_nodes} nodes cannot be incorporated into a spanning cycle.",
            "No Hamiltonian cycle possible. The best cycle covers {cycle_size}/{num_nodes} nodes. Interior nodes indicate structural obstacles to a Hamilton path.",
        ],
        'd_no': [
            "Not Hamiltonian, proven by cut-set analysis. Removing the {cut_size} BOLD center nodes creates {num_components} components, violating k(G-S) ≤ |S| since {num_components} > {cut_size}.",
            "The cut-set S (BOLD nodes) demonstrates non-Hamiltonicity. |S| = {cut_size} but k(G-S) = {num_components}, so k(G-S) > |S| proves no Hamiltonian cycle.",
            "Cut-set certificate: removing {cut_size} central BOLD nodes leaves {num_components} components. A Hamiltonian cycle cannot exist when k(G-S) > |S|.",
            "Non-Hamiltonicity proven. The {cut_size} cut-set nodes (BOLD) leave {num_components} isolated components when removed. This violates the necessary condition.",
        ],
    },
    
    'k_colorability': {
        'yes': [
            "The graph is {k}-colorable. The adjacency matrix shows exactly {k} empty diagonal blocks, each representing an independent set (color class) with no internal edges.",
            "{k}-colorable graph with {num_nodes} nodes. The {k} empty diagonal blocks are independent sets. Coloring each block with a distinct color produces a valid {k}-coloring.",
            "Valid {k}-coloring exists. The matrix structure reveals {k} independent sets forming empty diagonal blocks. All edges connect different color classes.",
            "The graph can be colored with {k} colors. Matrix analysis shows {k} blocks with sizes {block_sizes}. No two adjacent nodes share a color class.",
        ],
        's_no': [
            "The graph requires more than 4 colors. The matrix shows {actual_k} empty diagonal blocks - {actual_k} independent sets are needed, proving chromatic number > 4.",
            "Not 4-colorable. The {actual_k} diagonal blocks represent the minimum color classes required. Since {actual_k} > 4, four colors are insufficient.",
            "Chromatic number is {actual_k} (> 4). The matrix reveals {actual_k} independent sets are needed to partition the vertices without monochromatic edges.",
            "The graph needs {actual_k} colors, exceeding the 4-color limit. Each diagonal block is an independent set; fewer blocks would create intra-set edges.",
        ],
        'd_no': [
            "Contains a {clique_size}-clique (highlighted). These {clique_size} mutually adjacent vertices each require a different color, proving chromatic number ≥ {clique_size} > 4.",
            "The {clique_size}-clique in the BOLD top-left region proves χ(G) ≥ {clique_size}. All clique vertices are pairwise adjacent, requiring {clique_size} distinct colors.",
            "Not 4-colorable due to {clique_size}-clique. The highlighted complete subgraph forces {clique_size} colors minimum. The clique edges are shown in BOLD.",
            "A {clique_size}-clique is highlighted (BOLD edges). Since complete graphs on n vertices need n colors, this graph requires at least {clique_size} > 4 colors.",
        ],
    },
    
    'completeness': {
        'yes': [
            "This is a complete graph K_{num_nodes}. All {max_edges} possible edges are present (n(n-1)/2 for n={num_nodes}). The matrix is fully filled except the diagonal.",
            "Complete graph with {num_nodes} nodes. Every pair of distinct vertices is connected. The adjacency matrix is fully filled except the empty diagonal.",
            "K_{num_nodes} complete graph. All {max_edges} edges exist, meaning every vertex connects to every other vertex. The matrix confirms no missing off-diagonal entries.",
            "Completely connected graph. The {num_nodes} vertices form {max_edges} edges total. The full matrix (minus diagonal) proves this is a complete graph.",
        ],
        's_no': [
            "Not complete. The graph has {actual_edges} of {max_edges} possible edges ({completeness_percent}% complete). The empty cells in the matrix show {missing_count} missing edges.",
            "Incomplete graph - missing edges visible. Only {actual_edges}/{max_edges} edges present. The matrix shows empty gaps where edges should exist for completeness.",
            "This graph with {num_nodes} nodes has only {actual_edges} edges, not the required {max_edges}. Missing {missing_count} edges, visible as empty cells off-diagonal.",
            "Not a complete graph. Completeness requires {max_edges} edges but only {actual_edges} exist ({completeness_percent}%). Missing edges are marked by empty matrix cells.",
        ],
        'd_no': [
            "Non-complete graph. The nodes are sorted by degree (most to least connected). The top-left region is dense like a complete graph, but the bottom-right becomes increasingly sparse, revealing {missing_count} missing edges out of {max_edges} possible.",
            "Not complete — the degree-sorted matrix exposes the structure. The first few rows/columns are nearly complete, while the last nodes have very few connections, forming a near-independent set. Only {actual_edges}/{max_edges} edges present.",
            "The adjacency matrix sorts nodes from densest to sparsest. The transition from filled top-left to empty bottom-right visually proves incompleteness. {missing_count} vertex pairs lack edges.",
            "Degree-sorted matrix reveals incompleteness. Dense nodes (top-left) connect to almost everything, but sparse nodes (bottom-right) are barely connected. {completeness_percent}% of possible edges exist.",
        ],
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_cut_vertex_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for cut vertex reasoning."""
    info = {'num_components': 2, 'comp1_size': 0, 'comp2_size': 0, 'num_ears': 4}
    
    if isinstance(aux_data, list) and len(aux_data) > 0:
        # YES case: aux_data is list of cut vertices
        if 'component1_nodes' in G.graph:
            info['comp1_size'] = len(G.graph.get('component1_nodes', []))
            info['comp2_size'] = len(G.graph.get('component2_nodes', []))
    elif isinstance(aux_data, dict):
        # NO case: aux_data is dict with ears info
        if 'ears' in aux_data:
            info['num_ears'] = len(aux_data['ears'])
        elif 'layout_info' in aux_data:
            pass
    elif isinstance(aux_data, (list, tuple)) and len(aux_data) > 0:
        # d_no case: aux_data might be ears list
        if isinstance(aux_data[0], list):
            info['num_ears'] = len(aux_data)
    
    return info


def _extract_connectivity_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for connectivity reasoning."""
    components = list(nx.connected_components(G))
    sizes = sorted([len(c) for c in components], reverse=True)
    
    info = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_components': len(components),
        'largest_size': sizes[0] if sizes else 0,
        'isolated_nodes': sum(sizes[1:]) if len(sizes) > 1 else 0,
        'other_nodes': sum(sizes[1:]) if len(sizes) > 1 else 0,
        'component_sizes': ', '.join(map(str, sizes)),
    }
    return info


def _extract_bipartite_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for bipartite reasoning."""
    info = {
        'num_nodes': G.number_of_nodes(),
        'set_u_size': G.number_of_nodes() // 2,
        'set_v_size': G.number_of_nodes() - G.number_of_nodes() // 2,
        'cycle_length': 3,
        'cycle_description': 'nodes forming odd-length path',
    }
    
    if isinstance(aux_data, tuple) and len(aux_data) == 2:
        # YES case: (set_U, set_V)
        set_u, set_v = aux_data
        if hasattr(set_u, '__len__'):
            info['set_u_size'] = len(set_u)
            info['set_v_size'] = len(set_v)
    elif isinstance(aux_data, dict):
        if 'odd_cycle' in aux_data:
            cycle = aux_data['odd_cycle']
            info['cycle_length'] = len(cycle)
            info['cycle_description'] = f"nodes {cycle[:min(5, len(cycle))]}..."
    elif isinstance(aux_data, list) and len(aux_data) > 0:
        # Might be odd cycle
        info['cycle_length'] = len(aux_data)
        info['cycle_description'] = f"nodes {aux_data[:min(5, len(aux_data))]}..."
    
    return info


def _extract_planarity_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for planarity reasoning."""
    info = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'minor_type': 'K₅',
        'minor_size': 5,
    }
    
    if isinstance(aux_data, dict):
        if 'minor_info' in aux_data:
            minor_info = aux_data['minor_info']
            if isinstance(minor_info, tuple) and len(minor_info) >= 1:
                minor_type = minor_info[0]
                info['minor_type'] = minor_type
                info['minor_size'] = 5 if 'K5' in str(minor_type) else 6
    # aux_data is DIRECTLY the minor_info tuple for d_no
    elif isinstance(aux_data, tuple) and len(aux_data) >= 1:
        raw = str(aux_data[0])
        if 'K33' in raw:
            info['minor_type'] = 'K₃,₃'
            info['minor_size'] = 6
        else:
            info['minor_type'] = 'K₅'
            info['minor_size'] = 5
    
    return info


def _extract_outer_planarity_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for outer-planarity reasoning."""
    info = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'minor_type': 'K₄',
        'minor_size': 4,
    }
    
    if isinstance(aux_data, dict):
        if 'minor_info' in aux_data:
            minor_info = aux_data['minor_info']
            if isinstance(minor_info, tuple) and len(minor_info) >= 1:
                minor_type = minor_info[0]
                info['minor_type'] = str(minor_type).replace('K4', 'K₄').replace('K23', 'K₂,₃')
                info['minor_size'] = 4 if 'K4' in str(minor_type) else 5
    # aux_data is DIRECTLY the minor_info tuple for d_no
    elif isinstance(aux_data, tuple) and len(aux_data) >= 1:
        raw = str(aux_data[0])
        if 'K23' in raw:
            info['minor_type'] = 'K₂,₃'
            info['minor_size'] = 5
        else:
            info['minor_type'] = 'K₄'
            info['minor_size'] = 4
    
    return info


def _extract_hamiltonian_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for Hamiltonian reasoning."""
    info = {
        'num_nodes': G.number_of_nodes(),
        'cycle_size': G.number_of_nodes(),
        'interior_nodes': 0,
        'cut_size': 2,
        'num_components': 3,
    }
    
    if isinstance(aux_data, list):
        # YES case: aux_data is the Hamiltonian cycle
        info['cycle_size'] = len(aux_data)
    elif isinstance(aux_data, dict):
        if 'longest_cycle' in aux_data:
            cycle = aux_data['longest_cycle']
            if cycle:
                info['cycle_size'] = len(cycle)
                info['interior_nodes'] = G.number_of_nodes() - len(cycle)
        if 'cut_set_data' in aux_data:
            cut_set, components = aux_data['cut_set_data']
            info['cut_size'] = len(cut_set)
            info['num_components'] = len(components)
    # Direct (cut_set, components) tuple from d_no_info_key extraction
    elif isinstance(aux_data, tuple) and len(aux_data) == 2:
        cut_set, components = aux_data
        if isinstance(cut_set, set) and isinstance(components, list):
            info['cut_size'] = len(cut_set)
            info['num_components'] = len(components)
    
    return info


def _extract_k_colorability_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for k-colorability reasoning."""
    info = {
        'num_nodes': G.number_of_nodes(),
        'k': 4,
        'actual_k': 5,
        'clique_size': 5,
        'block_sizes': '5, 6, 5, 4',
    }
    
    if isinstance(aux_data, dict):
        # YES case: aux_data is coloring dict
        if all(isinstance(v, int) for v in aux_data.values()):
            num_colors = len(set(aux_data.values()))
            info['k'] = num_colors
            # Count block sizes
            from collections import Counter
            sizes = Counter(aux_data.values())
            info['block_sizes'] = ', '.join(map(str, sorted(sizes.values(), reverse=True)))
        elif 'coloring' in aux_data:
            coloring = aux_data['coloring']
            num_colors = len(set(coloring.values()))
            info['actual_k'] = num_colors
        elif 'clique_nodes' in aux_data:
            info['clique_size'] = len(aux_data['clique_nodes'])
    elif isinstance(aux_data, list):
        # d_no case: clique nodes
        info['clique_size'] = len(aux_data)
    
    return info


def _extract_completeness_info(G: nx.Graph, aux_data: Any) -> Dict[str, Any]:
    """Extract information for completeness reasoning."""
    n = G.number_of_nodes()
    max_edges = n * (n - 1) // 2
    actual_edges = G.number_of_edges()
    missing = max_edges - actual_edges
    
    info = {
        'num_nodes': n,
        'max_edges': max_edges,
        'actual_edges': actual_edges,
        'missing_count': missing,
        'completeness_percent': round(100 * actual_edges / max_edges, 1) if max_edges > 0 else 100,
    }
    
    return info


# Property to info extractor mapping
INFO_EXTRACTORS = {
    'cut_vertex': _extract_cut_vertex_info,
    'connectivity': _extract_connectivity_info,
    'bipartite': _extract_bipartite_info,
    'planarity': _extract_planarity_info,
    'outer_planarity': _extract_outer_planarity_info,
    'hamiltonian': _extract_hamiltonian_info,
    'k_colorability': _extract_k_colorability_info,
    'completeness': _extract_completeness_info,
}


# ============================================================================
# MAIN API
# ============================================================================

def generate_reasoning_text(
    property_name: str,
    label: str,  # 'yes', 's_no', or 'd_no'
    G: nx.Graph,
    auxiliary_data: Any = None
) -> str:
    """
    Generate detailed reasoning text for MLLM training.
    
    Args:
        property_name: One of the 8 graph properties
        label: 'yes', 's_no', or 'd_no'
        G: NetworkX graph
        auxiliary_data: Property-specific auxiliary data (e.g., cut vertices, cycle, etc.)
    
    Returns:
        Reasoning text explaining why the graph has/doesn't have the property
    """
    if property_name not in REASONING_TEMPLATES:
        return f"{'YES' if label == 'yes' else 'NO'}: Graph property {property_name} - {label}"
    
    templates = REASONING_TEMPLATES[property_name]
    
    if label not in templates:
        label = 'yes' if label == 'yes' else 's_no'  # Fallback
    
    # Get info extractor
    extractor = INFO_EXTRACTORS.get(property_name, lambda g, a: {})
    info = extractor(G, auxiliary_data)
    
    # Select random template
    template_list = templates.get(label, templates.get('yes', [""]))
    template = random.choice(template_list)
    
    # Format with extracted info
    try:
        text = template.format(**info)
    except KeyError:
        # If some keys are missing, use partial formatting
        text = template
        for key, value in info.items():
            text = text.replace('{' + key + '}', str(value))
    
    return text


def get_answer_with_reasoning(
    property_name: str,
    label: str,
    G: nx.Graph,
    auxiliary_data: Any = None,
    include_simple_answer: bool = True
) -> str:
    """
    Get a complete answer with reasoning for VQA-style training.
    
    Returns format:
        "YES/NO. [Detailed reasoning explanation]"
    """
    reasoning = generate_reasoning_text(property_name, label, G, auxiliary_data)
    return reasoning


# ============================================================================
# BATCH GENERATION UTILITIES
# ============================================================================

def generate_reasoning_batch(
    samples: List[Dict]
) -> List[str]:
    """
    Generate reasoning texts for a batch of samples.
    
    Args:
        samples: List of dicts with keys: 'property', 'label', 'graph', 'aux_data'
    
    Returns:
        List of reasoning texts
    """
    results = []
    for sample in samples:
        text = generate_reasoning_text(
            property_name=sample['property'],
            label=sample['label'],
            G=sample['graph'],
            auxiliary_data=sample.get('aux_data')
        )
        results.append(text)
    return results


if __name__ == '__main__':
    # Test the generator
    print("Testing Reasoning Text Generator...")
    
    # Create a simple test graph
    G = nx.cycle_graph(10)
    
    for prop in REASONING_TEMPLATES.keys():
        print(f"\n{'='*60}")
        print(f"Property: {prop}")
        print('='*60)
        
        for label in ['yes', 's_no', 'd_no']:
            text = generate_reasoning_text(prop, label, G, None)
            print(f"\n[{label.upper()}]:")
            print(f"  {text[:200]}...")
