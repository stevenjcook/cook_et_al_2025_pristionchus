"""
graphviz_loop.py
----------------

This script generates graphs using the NetworkX library. It takes two command-line arguments: the type of graph to generate and the path to the input file.

Usage:
    python graphviz_loop.py <GraphType> <InputFilePath>

Arguments:
    GraphType: The type of graph to generate. This also determines the name of the output directory. optiuons are 'full' or 'subset'
    InputFile: The path to the input file to use for generating the graph.

Example:
    python graphviz_loop.py full test_graphviz.csv

This script also creates an output directory in the same location as the script, with the name being the type of graph. If the directory already exists, it does not create a new one.

The script uses a color map for neurons and bins for histogram plotting.

Modules used:
    networkx, pandas, matplotlib, numpy, argparse, os

    NOTE: you will need to add the path to graphviz to your system PATH variable in order to use the 'circo' command for generating the graphs.
    
"""

import networkx as nx
import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from networkx.drawing.nx_pydot import graphviz_layout



# Create the parser
parser = argparse.ArgumentParser(description='Generate graphs.')

# Add the arguments
parser.add_argument('GraphType', type=str, help='The type of graph to generate')
parser.add_argument('InputFile', type=str, help='The input file to use for generating the graph')

# Parse the arguments
args = parser.parse_args()

script_dir = os.path.dirname(os.path.realpath(__file__))

output_dir = os.path.join(script_dir, args.GraphType)
os.makedirs(output_dir, exist_ok=True)

# List of neurons
color_map = {'not specific': 'gray', 'core': 'black', 'pristi_specific': 'blue', 'herm_specific': 'red'}
bins = [0, 3, 10, np.inf]
labels = ['0.2', '2', '5']



# Import edge list from .csv files
df = pd.read_csv(args.InputFile)
df['weight_bins'] = pd.cut(df['average_synaptic_weight'], bins=bins, labels=labels)
df_subset = df[df['species_specific'] != 'not specific']


#get all unique neuron names from the columns 'pre' and 'post'
neurons = pd.unique(df[['pre', 'post']].values.ravel('K'))

# Create a directed graph from edge list
G_full = nx.from_pandas_edgelist(df, source='pre', target='post', edge_attr=['species_specific', 'average_synaptic_weight'], create_using=nx.DiGraph())
G_subset = nx.from_pandas_edgelist(df_subset, source='pre', target='post', edge_attr=['species_specific', 'average_synaptic_weight'], create_using=nx.DiGraph())

if args.GraphType == 'subset':
    G = G_subset
elif args.GraphType == 'full':
    G = G_full

# Iterate over neurons
for neuron in neurons:
    # Create a new graph for the subgraph
    subgraph = nx.DiGraph()
    
    # Add the neuron to the subgraph
    subgraph.add_node(neuron)
    outputs = []
    inputs = []
    
    # Loop over the edges that the neuron makes
    for source, target in G.out_edges(neuron):
        # Ensure the source is the neuron we are interested in
        if source == neuron:
            # Get the attributes of the edge
            edge_attr = G[source][target]
            
            # Add the edge to the subgraph
            color = color_map[edge_attr['species_specific']]
            linewidth = df.loc[(df['pre'] == source) & (df['post'] == target), 'weight_bins'].values[0]
            subgraph.add_edge(source, target, test1=edge_attr['species_specific'], color=color, penwidth=linewidth, arrowsize='1')
            outputs.append(target)

        # Ensure the source is the neuron we are interested in
    for source, target in G.in_edges(neuron):
        if target == neuron:
            # Get the attributes of the edge
            edge_attr = G[source][target]
            
            # Add the edge to the subgraph
            color = color_map[edge_attr['species_specific']]
            linewidth = df.loc[(df['pre'] == source) & (df['post'] == target), 'weight_bins'].values[0]
            subgraph.add_edge(source, target, test1=edge_attr['species_specific'], color=color, penwidth=linewidth, arrowsize='1')
            inputs.append(target)
    
    # Convert NetworkX graph to a PyGraphviz graph
    A = to_agraph(subgraph)
    A.graph_attr['splines'] = 'polyline'
    A.graph_attr['overlap'] = 'false'
    A.graph_attr['ranksep'] = '0.5'
    A.graph_attr['nodesep'] = '0.6'
    A.graph_attr['rankdir'] = 'LR'
    A.get_node(neuron).attr['width'] = '3'
    A.get_node(neuron).attr['height'] = '3'
    A.get_node(neuron).attr['fontsize'] = '50'


    # Set the positions of the nodes
    A.layout(prog='circo')
    A.draw(os.path.join(output_dir, f'{neuron}.svg'), prog='dot')
    #A.write(os.path.join(output_dir, f'{neuron}.dot'))


import cairosvg
import PyPDF2

# Convert SVG files to PDF
for neuron in neurons:
    svg_file = os.path.join(output_dir, f'{neuron}.svg')
    pdf_file = os.path.join(output_dir, f'{neuron}.pdf')
    if os.path.exists(svg_file):
        cairosvg.svg2pdf(url=svg_file, write_to=pdf_file)

# Merge PDF files
merger = PyPDF2.PdfMerger()

for neuron in neurons:
    pdf_file = os.path.join(output_dir, f'{neuron}.pdf')
    if os.path.exists(pdf_file):
        merger.append(pdf_file)

# Write the merged PDF to a file
merged_pdf = os.path.join(output_dir, 'merged.pdf')
merger.write(merged_pdf)
merger.close()