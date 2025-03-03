import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
import pickle

graphviz_path = "C:\\Program Files\\Graphviz\\bin"
os.environ["PATH"] += os.pathsep + graphviz_path

df = pd.read_csv('c:/Users/steve/Documents/GitHub/cook_et_al_2024_pristionchus/circuit_diagrams/directed_species_supplemental_class1.csv')

droplist = ['HSN', 'VBn', 'AVM', 'EFn', 'MCM', 'CEM', 'VCn', 'DBn', 'CAN', 'SAB']
df = df[~df.pre.isin(droplist)]
df = df[~df.post.isin(droplist)].reset_index(drop=True)


print(len(df))
# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def calculate_positions(G, layout_type, seed=42):
    """
    Calculate node positions using specified layout algorithm.
    """
    print(f"Calculating {layout_type} layout...")
    
    if layout_type == 'spring':
        k = 10
        return nx.spring_layout(G, seed=seed, k=k, iterations=5, weight='weight')
    elif layout_type == 'circular':
        return nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        return nx.kamada_kawai_layout(G)
    elif layout_type == 'shell':
        return nx.shell_layout(G)
    elif layout_type == 'spectral':
        return nx.arf_layout(G)
    elif layout_type == 'graphviz_dot':
        try:
            print("Graphviz layout")
            return nx.nx_agraph.graphviz_layout(G, prog='dot', overlap=False)
        except:
            print("Graphviz not available, falling back to spring layout")
            return nx.spring_layout(G, seed=seed)
    else:
        return nx.spring_layout(G, seed=seed)

def save_positions(positions, filename='node_positions.pkl'):
    """Save node positions to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(positions, f)
    print(f"Positions saved to {filename}")

def load_positions(filename='node_positions.pkl'):
    """Load node positions from a file"""
    try:
        with open(filename, 'rb') as f:
            positions = pickle.load(f)
        print(f"Positions loaded from {filename}")
        return positions
    except FileNotFoundError:
        print(f"Position file {filename} not found")
        return None

def create_network_subplots(df, source_col='pre', target_col='post', 
                           species_col='species_specific',
                           weight_col='average_synaptic_weight',
                           positions=None,
                           save_pos=False,
                           layout_type='graphviz_dot',
                           node_size_multiplier=3.0):
    """
    Create a figure with subplots for different connection types.
    """
    # Check if weight column exists
    use_weights = weight_col in df.columns
    
    # Create the graph
    if use_weights:
        G = nx.from_pandas_edgelist(
            df, 
            source=source_col,
            target=target_col,
            edge_attr=[weight_col],
            create_using=nx.DiGraph()
        )
        print(f"Using weight column: {weight_col}")
        # Debug: print some weight values
        edge_data = list(G.edges(data=True))[:5]
        print(f"Sample edge data: {edge_data}")
    else:
        # Create graph without weights if column doesn't exist
        G = nx.from_pandas_edgelist(
            df, 
            source=source_col,
            target=target_col,
            create_using=nx.DiGraph()
        )
        print(f"Warning: Weight column '{weight_col}' not found in DataFrame. Using uniform edge weights.")
    
    # Remove self-loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print(f"Removed {len(self_loops)} self-loops from the graph.")
    print(f"Graph now has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Bin edge weights into 4 categories or use default if no weights
    if use_weights:
        weight_values = [data.get(weight_col, 1) for _, _, data in G.edges(data=True)]
        print(f"Number of edges with weights: {len(weight_values)}")
        print(f"Weight range: {min(weight_values)} to {max(weight_values)}")
        
        if weight_values:
            # Define weight bins
            q1, q2, q3 = np.percentile(weight_values, [25, 50, 75])
            print(f"Weight quartiles: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
            
            # Map weights to width bins - using very distinct width values
            def get_width_from_weight(weight):
                if weight <= q1:
                    return 0.1, 0  # Width, bin index
                elif weight <= q2:
                    return 0.25, 1
                elif weight <= q3:
                    return 0.5, 2
                else:
                    return 2.0, 3
        else:
            # Fallback if no weights
            def get_width_from_weight(weight):
                return 2.0, 0
    else:
        # If weight column doesn't exist, use uniform width
        def get_width_from_weight(weight):
            return 2.0, 0
        # Set dummy values for legend
        q1, q2, q3 = 0.25, 0.5, 0.75
        weight_values = [1]  # Dummy list so the legend code doesn't break
    
    # Calculate edge widths dictionary
    edge_widths = {}
    edge_bins = {}
    for u, v, data in G.edges(data=True):
        if use_weights:
            # Make sure we're accessing the weight correctly
            if weight_col in data:
                weight = data[weight_col]
            else:
                weight = 1
                print(f"Warning: Edge {u}->{v} doesn't have weight attribute")
        else:
            weight = 1
        width, bin_idx = get_width_from_weight(weight)
        edge_widths[(u, v)] = width
        edge_bins[(u, v)] = bin_idx
    
    # Debug: print some samples from the edge_widths dictionary
    sample_widths = list(edge_widths.items())[:5]
    print(f"Sample edge widths: {sample_widths}")
    
    # Get or calculate node positions
    if positions is None:
        # First try to load saved positions
        #positions = load_positions()
        
        # If loading failed, calculate new positions
        if positions is None:
            positions = calculate_positions(G, layout_type=layout_type)
            
            # Save positions if requested
            if save_pos:
                save_positions(positions)
    
    # Categorize edges by species
    pacificus_edges = []
    elegans_edges = []
    core_edges = []
    other_edges = []
    
    for _, row in df.iterrows():
        source = row[source_col]
        target = row[target_col]
        species_type = row[species_col]
        
        if species_type == 'pristi_specific':
            pacificus_edges.append((source, target))
        elif species_type == 'cel_specific':
            elegans_edges.append((source, target))
        elif species_type == 'core':
            core_edges.append((source, target))
        else:
            other_edges.append((source, target))
    
    # Verify all edges are being captured
    print(f"Total edges in DataFrame: {len(df)}")
    print(f"Categorized edges: P. pacificus: {len(pacificus_edges)}, C. elegans: {len(elegans_edges)}, Core: {len(core_edges)}, Other: {len(other_edges)}")
    print(f"Total categorized: {len(pacificus_edges) + len(elegans_edges) + len(core_edges) + len(other_edges)}")
    
    # Identify nodes involved in each type of connection
    pacificus_nodes = set()
    for s, t in pacificus_edges:
        pacificus_nodes.add(s)
        pacificus_nodes.add(t)
        
    elegans_nodes = set()
    for s, t in elegans_edges:
        elegans_nodes.add(s)
        elegans_nodes.add(t)
        
    core_nodes = set()
    for s, t in core_edges:
        core_nodes.add(s)
        core_nodes.add(t)
    
    # Calculate node sizes - INCREASED base size with multiplier
    node_sizes = {node: 600 for node in G.nodes()}
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    subplot_titles = [
        '$\it{C. elegans}$-Specific Connections',
        '$\it{P. pacificus}$-Specific Connections',
        'Core Connections',
        'All Connections'
    ]
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # 1. C. elegans Specific (top-left)
    ax = axes[0]
    # Add subplot label 'A'
    ax.text(-0.1, 1.05, 'A', transform=ax.transAxes, 
            fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Draw background edges with varying widths
    for edge in pacificus_edges + core_edges + other_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='lightgray',
                                width=edge_widths[edge] * 0.5,  # Half-width for background
                                alpha=0.4,
                                arrowsize=15,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    # Draw highlighted edges with varying widths
    for edge in elegans_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='red',
                                width=edge_widths[edge],
                                alpha=1.0,
                                arrowsize=20,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    # Draw nodes
    node_colors = ['lightcoral' if node in elegans_nodes else 'lightgray' for node in G.nodes()]
    edge_colors = ['red' if node in elegans_nodes else 'gray' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, positions,
                         node_color=node_colors,
                         node_size=list(node_sizes.values()),
                         alpha=1.0,
                         edgecolors=edge_colors,
                         linewidths=1.5,  # Increased linewidth
                         ax=ax)
    
    # Add labels with larger font
    nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(subplot_titles[0], fontsize=18)
    ax.axis('off')
    
    # 2. P. pacificus Specific (top-right)
    ax = axes[1]
    # Add subplot label 'B'
    ax.text(-0.1, 1.05, 'B', transform=ax.transAxes, 
            fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Draw background edges with varying widths
    for edge in elegans_edges + core_edges + other_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='lightgray',
                                width=edge_widths[edge] * 0.5,
                                alpha=0.3,
                                arrowsize=15,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    # Draw highlighted edges with varying widths
    for edge in pacificus_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='blue',
                                width=edge_widths[edge],
                                alpha=1.0,
                                arrowsize=20,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    # Draw nodes
    node_colors = ['lightblue' if node in pacificus_nodes else 'lightgray' for node in G.nodes()]
    edge_colors = ['blue' if node in pacificus_nodes else 'gray' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, positions,
                         node_color=node_colors,
                         node_size=list(node_sizes.values()),
                         alpha=1.0,
                         edgecolors=edge_colors,
                         linewidths=.0,
                         ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(subplot_titles[1], fontsize=18)
    ax.axis('off')
    
    # 3. Core Connections (bottom-left)
    ax = axes[2]
    # Add subplot label 'C'
    ax.text(-0.1, 1.05, 'C', transform=ax.transAxes, 
            fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Draw background edges with varying widths
    for edge in elegans_edges + pacificus_edges + other_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='lightgray',
                                width=edge_widths[edge] * 0.5,
                                alpha=0.4,
                                arrowsize=15,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    # Draw highlighted edges with varying widths
    for edge in core_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='black',
                                width=edge_widths[edge],
                                alpha=1.0,
                                arrowsize=20,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    # Draw nodes
    node_colors = ['gray' if node in core_nodes else 'lightgray' for node in G.nodes()]
    edge_colors = ['black' if node in core_nodes else 'lightgray' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, positions,
                         node_color=node_colors,
                         node_size=list(node_sizes.values()),
                         alpha=1.0,
                         edgecolors=edge_colors,
                         linewidths=1.0,
                         ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(subplot_titles[2], fontsize=18)
    ax.axis('off')
    
    # 4. All Connections (bottom-right)
    ax = axes[3]
    # Add subplot label 'D'
    ax.text(-0.1, 1.05, 'D', transform=ax.transAxes, 
            fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Draw all edges with appropriate colors and varying widths
    for edge in elegans_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='gray',
                                width=edge_widths[edge],
                                alpha=0.4,
                                arrowsize=15,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    for edge in pacificus_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='gray',
                                width=edge_widths[edge],
                                alpha=0.4,
                                arrowsize=15,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    for edge in core_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='gray',
                                width=edge_widths[edge],
                                alpha=0.4,
                                arrowsize=15,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    for edge in other_edges:
        if edge in edge_widths:
            nx.draw_networkx_edges(G, positions, 
                                edgelist=[edge],
                                edge_color='gray',
                                width=edge_widths[edge] * 0.5,
                                alpha=0.4,
                                arrowsize=10,
                                ax=ax,
                                node_size=list(node_sizes.values()))
    
    # Draw nodes with uniform coloring for the "All Connections" plot
    nx.draw_networkx_nodes(G, positions,
                        node_color='lightgray',  # Same neutral color for all nodes
                        node_size=list(node_sizes.values()),
                        alpha=1.0,
                        edgecolors='gray',  # Uniform edge color
                        linewidths=1.0,
                        ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(subplot_titles[3], fontsize=18)
    ax.axis('off')
    
    # Add a legend for connection types to the second plot (top-right)
    axes[1].legend([
        plt.Line2D([0], [0], color='red', lw=2),
        plt.Line2D([0], [0], color='blue', lw=2),
        plt.Line2D([0], [0], color='black', lw=2),
        plt.Line2D([0], [0], color='grey', lw=2, alpha=0.25)
    ], 
    ["$\it{C. elegans}$-specific", "$\it{P. pacificus}$-specific",
     "Core", "Other"], 
    title='Connection Type', loc='lower right', fontsize='10', title_fontsize='12')
    
    # Add a legend for edge weights to the third plot (bottom-left)
    if weight_values:
        axes[2].legend([
            plt.Line2D([0], [0], color='black', lw=0.1),
            plt.Line2D([0], [0], color='black', lw=0.25),
            plt.Line2D([0], [0], color='black', lw=0.5),
            plt.Line2D([0], [0], color='black', lw=2.0),
        ], 
        [f" Quartile 1", f"Quartile 2", f"Quartile 3", f"Quartile 4"], 
        title=f'Synaptic Weight', loc='lower right', fontsize='10', title_fontsize='12')
    
    # Add overall title
    #plt.suptitle("Neural Network Connections by Species", fontsize=24, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save figure
    plt.savefig('output/network_connections_subplots.png', dpi=300, bbox_inches='tight')
    plt.savefig('output/network_connections_subplots.svg', format='svg', bbox_inches='tight')
    
    return G, positions

# Example usage
if __name__ == "__main__":
    # We've already loaded the real data at the top of the file
    # Don't create a synthetic dataset that overrides it
    
    # The real data is already in the 'df' variable from this line:
    # df = pd.read_csv('c:/Users/steve/Documents/GitHub/cook_et_al_2024_pristionchus/circuit_diagrams/directed_species_supplemental_class1.csv')
    
    print(f"Using the dataframe with {len(df)} edges")
    
    # Create the subplots with larger nodes
    G, positions = create_network_subplots(
        df, 
        save_pos=True, 
        layout_type='spring',
        node_size_multiplier=0.5  # Adjust this value to make nodes larger/smaller
    )
    
    print("All subplots created with fixed node positions!")
    plt.show()  # Display the figure