from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np
from scipy import stats, sparse
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
df_merged2 = pd.read_csv(current_dir + "/output/adj_syn_merged_sjc.csv")

def analyze_species_specific(df, random_df, columns_species1, columns_species2):
    """Analyze species-specific patterns in observed and random data"""
    # Observed counts
    observed_species1 = len(df[
        (df[columns_species1].astype(bool).sum(axis=1) == len(columns_species1)) & 
        (df[columns_species2].astype(bool).sum(axis=1) == 0)
    ])
    observed_species2 = len(df[
        (df[columns_species1].astype(bool).sum(axis=1) == 0) & 
        (df[columns_species2].astype(bool).sum(axis=1) == len(columns_species2))
    ])
    
    # Random counts
    random_species1 = len(random_df[
        (random_df[columns_species1].astype(bool).sum(axis=1) == len(columns_species1)) & 
        (random_df[columns_species2].astype(bool).sum(axis=1) == 0)
    ])
    random_species2 = len(random_df[
        (random_df[columns_species1].astype(bool).sum(axis=1) == 0) & 
        (random_df[columns_species2].astype(bool).sum(axis=1) == len(columns_species2))
    ])
    
    return {
        'observed_species1': observed_species1,
        'observed_species2': observed_species2,
        'random_species1': random_species1,
        'random_species2': random_species2
    }

def randomize_single_network(args):
    """Single network randomization function for parallel processing"""
    G, n_swaps = args
    G_random = G.copy()
    try:
        nx.double_edge_swap(G_random, nswap=n_swaps)
    except nx.NetworkXError:
        pass
    return list(G_random.edges())

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np
from scipy import stats, sparse
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

def randomize_single_iteration(args):
    """Process a single randomization iteration"""
    matrices, df, columns_species1, columns_species2, pre_neurons, post_neurons = args
    
    # Randomize each matrix while preserving degree distribution
    random_matrices = []
    for mat in matrices:
        rows, cols = mat.nonzero()
        edges = np.array(list(zip(rows, cols)))
        n_edges = len(edges)
        
        # Vectorized edge swapping
        if n_edges >= 2:
            n_swaps = n_edges * 5
            for _ in range(n_swaps):
                idx1, idx2 = np.random.choice(n_edges, 2, replace=False)
                edge1, edge2 = edges[idx1], edges[idx2]
                
                new_edge1 = (edge1[0], edge2[1])
                new_edge2 = (edge2[0], edge1[1])
                
                if (new_edge1[0] != new_edge1[1] and new_edge2[0] != new_edge2[1]):
                    edges[idx1] = new_edge1
                    edges[idx2] = new_edge2
        
        if len(edges) > 0:
            random_mat = sparse.csr_matrix(
                (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                shape=mat.shape
            )
        else:
            random_mat = sparse.csr_matrix(mat.shape)
        
        random_matrices.append(random_mat)
    
    # Convert to dataframe format efficiently
    random_df = pd.DataFrame(index=df.index)
    for mat, col in zip(random_matrices, columns_species1 + columns_species2):
        random_df[col] = 0
        rows, cols = mat.nonzero()
        edges = [(pre_neurons[r], post_neurons[c]) for r, c in zip(rows, cols)]
        if edges:
            edge_df = pd.DataFrame(edges, columns=['pre', 'post'])
            for _, edge in edge_df.iterrows():
                mask = (df['pre'] == edge['pre']) & (df['post'] == edge['post'])
                random_df.loc[mask, col] = 1
    
    # Calculate distributions and counts
    random_counts = random_df[columns_species1 + columns_species2].astype(bool).sum(axis=1)
    counts_distribution = random_counts.value_counts().sort_index()
    
    # Analyze species-specific patterns
    species_counts = analyze_species_specific(df, random_df, columns_species1, columns_species2)
    
    return counts_distribution, species_counts

def fast_randomize_connections(df, columns_species1, columns_species2, n_iterations=1000):
    """Optimized randomization using sparse matrices and parallel processing"""
    # Use all available cores except one
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # Create neuron to index mappings once
    pre_neurons = sorted(df['pre'].unique())
    post_neurons = sorted(df['post'].unique())
    pre_to_idx = {neuron: i for i, neuron in enumerate(pre_neurons)}
    post_to_idx = {neuron: i for i, neuron in enumerate(post_neurons)}
    
    # Convert each column to sparse matrix once
    matrices = []
    for col in columns_species1 + columns_species2:
        edges = df[df[col] != 0][['pre', 'post']]
        rows = [pre_to_idx[pre] for pre in edges['pre']]
        cols = [post_to_idx[post] for post in edges['post']]
        mat = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(pre_neurons), len(post_neurons))
        )
        matrices.append(mat)
    
    # Prepare arguments for parallel processing
    args_list = [(matrices, df, columns_species1, columns_species2, pre_neurons, post_neurons)
                 for _ in range(n_iterations)]
    
    # Process in parallel
    random_counts_distribution = []
    species_specific_counts = []
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        print(f"\nUsing {n_cores} CPU cores for parallel processing")
        futures = [executor.submit(randomize_single_iteration, args) for args in args_list]
        
        for future in tqdm(futures, total=n_iterations, desc="Processing iterations"):
            counts_dist, species_counts = future.result()
            random_counts_distribution.append(counts_dist)
            species_specific_counts.append(species_counts)
    
    # Get observed distribution
    observed_counts = df[columns_species1 + columns_species2].astype(bool).sum(axis=1)
    
    return random_counts_distribution, observed_counts, species_specific_counts

def compare_random_distributions(df_adj, df_syn, n_iterations=1000):
    """Compare observed patterns to random distributions"""
    # Define column lists for both adjacency and synaptic data
    columns_elegans_adj = ['witvliet_6', 'witvliet_4', 'witvliet_3', 'witvliet_2', 'witvliet_1', 
                          'cel_jsh', 'cel_n2u', 'witvliet_8', 'witvliet_5']
    columns_pristi_adj = ['pristi_s14', 'pristi_s15']
    
    columns_elegans_syn = ['witvliet_6_syn', 'witvliet_4_syn', 'witvliet_3_syn', 'witvliet_2_syn', 'witvliet_1_syn', 
                          'cel_jsh_syn', 'cel_n2u_syn', 'witvliet_8_syn', 'witvliet_5_syn', 'witvliet_7_syn']
    columns_pristi_syn = ['pristi_s14_syn', 'pristi_s15_syn']
    
    print("Generating random distributions...")
    
    print("\nAnalyzing synaptic connectivity...")
    random_syn_dist, observed_syn, syn_species_counts = fast_randomize_connections(
        df_syn, columns_elegans_syn, columns_pristi_syn, n_iterations)
    
    print("\nAnalyzing adjacency...")
    random_adj_dist, observed_adj, adj_species_counts = fast_randomize_connections(
        df_adj, columns_elegans_adj, columns_pristi_adj, n_iterations)
    
    # Calculate statistics
    def calculate_stats(observed, random_dist):
        z_score = (observed - np.mean(random_dist)) / np.std(random_dist)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        return z_score, p_value, np.mean(random_dist), np.std(random_dist)
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 1.25})
    
    # Plot synaptic connectivity distribution
    syn_df = pd.DataFrame(random_syn_dist)
    syn_df = syn_df.fillna(0)
    syn_mean = syn_df.mean()
    syn_std = syn_df.std()
    
    ax1.errorbar(syn_mean.index, syn_mean, yerr=syn_std, 
                color='gray', alpha=0.5, label='Random (mean ± std)')
    observed_syn_counts = observed_syn.value_counts().sort_index()
    ax1.plot(observed_syn_counts.index, observed_syn_counts, 
             'r-', label='Observed', linewidth=2)
    ax1.set_xlabel('Number of Datasets')
    ax1.set_ylabel('Count of Connections')
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax1.legend(fontsize = 10)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold')
    
    # Plot adjacency distribution
    adj_df = pd.DataFrame(random_adj_dist)
    adj_df = adj_df.fillna(0)
    adj_mean = adj_df.mean()
    adj_std = adj_df.std()
    
    ax2.errorbar(adj_mean.index, adj_mean, yerr=adj_std, 
                color='gray', alpha=0.5, label='Random (mean ± std)')
    observed_adj_counts = observed_adj.value_counts().sort_index()
    ax2.plot(observed_adj_counts.index, observed_adj_counts, 
             'r-', label='Observed', linewidth=2)
    ax2.set_xlabel('Number of Datasets')
    ax2.set_ylabel('Count of Adjacencies')
    ax2.legend(fontsize = 10)
    ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold')
    
    # Plot species-specific comparisons with statistics
    syn_species_df = pd.DataFrame(syn_species_counts)
    ax3.boxplot([syn_species_df['random_species1'], syn_species_df['random_species2']], 
                labels=['$\it{C. elegans}$', '$\it{P. pacificus}$'])
    ax3.scatter([1, 2], 
                [syn_species_df['observed_species1'].iloc[0], 
                 syn_species_df['observed_species2'].iloc[0]], 
                color='red', marker='*', s=100, label='Observed')
    
    # Add statistics for synaptic connectivity
    stats_text = []
    for i, (species, col) in enumerate([('C. elegans', 'random_species1'), 
                                      ('P. pacificus', 'random_species2')]):
        z, p, mean, std = calculate_stats(
            syn_species_df[col.replace('random', 'observed')].iloc[0],
            syn_species_df[col]
        )
        stats_text.append(
            f"{species}\n"
            f"Z = {z:.2f}\n"
            f"p = {p:.2e}\n"
            f"Random: {mean:.1f}±{std:.1f}\n"
            f"Obs: {syn_species_df[col.replace('random', 'observed')].iloc[0]}"
        )
    
    # Add vertical line between conditions
    ax3.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add statistics text
    # ax3.text(0.5, ax3.get_ylim()[1], stats_text[0], 
    #          verticalalignment='top', horizontalalignment='right', fontsize = 6)
    # ax3.text(2, ax3.get_ylim()[1], stats_text[1], 
    #          verticalalignment='top', horizontalalignment='left', fontsize = 6)
    
    ax3.set_ylabel('Count of Connections')
    ax3.legend(loc = 'center left', fontsize = 10)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=14, fontweight='bold')
    
    # Repeat for adjacency plot
    adj_species_df = pd.DataFrame(adj_species_counts)
    ax4.boxplot([adj_species_df['random_species1'], adj_species_df['random_species2']], 
                labels=['$\it{C. elegans}$', '$\it{P. pacificus}$'])
    ax4.scatter([1, 2], 
                [adj_species_df['observed_species1'].iloc[0], 
                 adj_species_df['observed_species2'].iloc[0]], 
                color='red', marker='*', s=100, label='Observed')
    
    # Add statistics for adjacency
    stats_text = []
    for i, (species, col) in enumerate([('C. elegans', 'random_species1'), 
                                      ('P. pacificus', 'random_species2')]):
        z, p, mean, std = calculate_stats(
            adj_species_df[col.replace('random', 'observed')].iloc[0],
            adj_species_df[col]
        )
        stats_text.append(
            f"{species}\n"
            f"Z = {z:.2f}\n"
            f"p = {p:.2e}\n"
            f"Random: {mean:.1f}±{std:.1f}\n"
            f"Obs: {adj_species_df[col.replace('random', 'observed')].iloc[0]}"
        )
    
    # # Add vertical line between conditions
    # ax4.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add statistics text
    # ax4.text(.8, ax4.get_ylim()[1], stats_text[0], 
    #          verticalalignment='top', horizontalalignment='right', fontsize=6)
    # ax4.text(1.8, ax4.get_ylim()[1], stats_text[1], 
    #          verticalalignment='top', horizontalalignment='left', fontsize=6)
    
    ax4.set_ylabel('Count of Adjacencies')
    ax4.legend(loc = 'center left', fontsize = 10)
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(current_dir + "/output/random_distribution_comparison.svg")
    plt.savefig(current_dir + "/output/random_distribution_comparison.png")
    plt.show()

# Run the analysis
if __name__ == '__main__':
    # Load both dataframes
    df_adj = pd.read_csv(current_dir + "/output/adj_syn_merged_sjc.csv")
    df_syn = pd.read_csv(current_dir + "/output/fullsynapses_sjc.csv")
    
    compare_random_distributions(df_adj, df_syn, n_iterations=1)