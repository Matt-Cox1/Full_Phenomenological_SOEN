# FILENAME: utils/soen_model_utils.py


import sys
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
from matplotlib.patches import Rectangle
from typing import Literal, Union, Optional

from model.model_config import SOENConfig
from model.soen_model import *







from matplotlib.patches import Rectangle
from matplotlib.legend import Legend



def visualise_soen_network(
    model: 'SOENModel',
    dpi: int = 300,
    show_labels: bool = False,
    dark_mode: bool = False,
    node_size: int = 500,
    figsize: tuple[int, int] = (18, 8),
    line_thickness: int = 5,
    legend_fontsize: int = 10,
    layout: Literal['circular', 'linear'] = 'circular',
    show_legend: bool = True,
) -> None:
    # Create figure and axes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]

    edge_threshold = 0
    G = nx.DiGraph()

    # Color scheme setup
    if dark_mode:
        bg_color, text_color = '#1e1e1e', 'white'
        input_color, hidden_color, output_color = '#FF6B6B', '#4ECDC4', '#FFA500'
        line_color, border_color, node_edge_color = '#555555', 'white', 'white'
    else:
        bg_color, text_color = 'white', 'black'
        input_color, hidden_color, output_color = '#FF6B6B', '#20639B', '#FFA500'
        line_color, border_color, node_edge_color = '#CCCCCC', 'black', 'black'

    # Modified node setup for multiple hidden layers
    input_nodes = [f'input_{i}' for i in range(model.num_input)]
    hidden_layers = []
    total_hidden_nodes = 0
    for layer_idx, layer_size in enumerate(model.num_hidden):
        layer_nodes = [f'hidden_{layer_idx}_{i}' for i in range(layer_size)]
        hidden_layers.append(layer_nodes)
        total_hidden_nodes += layer_size
    output_nodes = [f'output_{i}' for i in range(model.num_output)]

    # Create flattened list of all nodes
    all_nodes = input_nodes + [node for layer in hidden_layers for node in layer] + output_nodes

    # Add nodes to graph with appropriate colors
    all_layers = [input_nodes] + hidden_layers + [output_nodes]
    num_layers = len(all_layers)
    
    # Generate colors for hidden layers (gradient between hidden_color and output_color)
    if len(hidden_layers) > 1:
        hidden_colors = [
            _interpolate_color(hidden_color, output_color, i/(len(hidden_layers)-1))
            for i in range(len(hidden_layers))
        ]
    else:
        hidden_colors = [hidden_color]

    # Add nodes with colors
    for layer_idx, layer_nodes in enumerate(all_layers):
        if layer_idx == 0:
            color = input_color
        elif layer_idx == num_layers - 1:
            color = output_color
        else:
            color = hidden_colors[layer_idx - 1]
        
        if layer_nodes:
            G.add_nodes_from(layer_nodes, color=color, layer=layer_idx)

    # Optimize J matrix operations - do this once upfront
    J_masked = model.J.data.cpu().numpy() * model.mask.cpu().numpy()
    edge_indices = np.where(np.abs(J_masked) > edge_threshold)
    
    # Modified edge setup - much faster than nested loops
    for from_idx, to_idx in zip(*edge_indices):
        from_node = all_nodes[from_idx]
        to_node = all_nodes[to_idx]
        weight = abs(J_masked[from_idx, to_idx])
        sign = np.sign(J_masked[from_idx, to_idx])
        G.add_edge(from_node, to_node, weight=weight, sign=sign)

    # Modified layout calculation
    active_layers = [layer for layer in all_layers if layer]
    num_active_layers = len(active_layers)
    layer_positions = [i / (num_active_layers - 1) if num_active_layers > 1 else 0.5 
                      for i in range(num_active_layers)]

    # Custom layout
    pos = {}
    if layout == 'linear':
        for layer_nodes, x_pos in zip(active_layers, layer_positions):
            layer_height = len(layer_nodes)
            for j, node in enumerate(layer_nodes):
                y = (j - (layer_height - 1) / 2) / max(layer_height - 1, 1)
                pos[node] = (x_pos, y)
    elif layout == 'circular':
        for layer_nodes, x_pos in zip(active_layers, layer_positions):
            if len(layer_nodes) == 1:
                pos.update({node: (x_pos, 0) for node in layer_nodes})
            elif len(layer_nodes) == 2:
                pos.update({node: (x_pos, j-0.5) for j, node in enumerate(layer_nodes)})
            else:
                radius = 0.2
                for j, node in enumerate(layer_nodes):
                    angle = 2 * np.pi * j / len(layer_nodes)
                    pos[node] = (x_pos + radius * np.cos(angle), radius * np.sin(angle))
    else:
        raise ValueError("Invalid layout option. Choose 'linear' or 'circular'.")

    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, edgecolors=node_edge_color, ax=ax)

    # Draw edges
    if G.edges():
        max_weight = max(data['weight'] for (_, _, data) in G.edges(data=True))
        for (u, v, data) in G.edges(data=True):
            edge_style = 'solid' if data['sign'] > 0 else 'dashed'
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=node_edge_color, 
                                   width=(0.3 + data['weight'] * line_thickness / max_weight), 
                                   alpha=0.9, ax=ax, style=edge_style, arrows=True, 
                                   arrowsize=10, arrowstyle='->')

    # Labels and styling
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, font_color=text_color, ax=ax)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')

    # Border
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=border_color, linewidth=2, transform=ax.transAxes))

    # Key Banner
    legend_elements = []
    for nodes, color, label in zip([input_nodes, hidden_layers, output_nodes],
                                   [input_color, hidden_colors, output_color],
                                   ['Input Node', 'Hidden Node', 'Output Node']):
        if nodes:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label, 
                                              markerfacecolor=color, markersize=15))
    if G.edges():
        legend_elements.extend([
            plt.Line2D([0], [0], color=node_edge_color, lw=2, label='Excitatory Connection'),
            plt.Line2D([0], [0], color=node_edge_color, lw=2, linestyle='dashed', label='Inhibitory Connection')
        ])
    
    # Create a separate axes for the legend
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.08], facecolor='none')  # Adjust the position and size as needed
    legend_ax.axis('off')

    # Create the legend with the input font size
    legend = Legend(legend_ax, legend_elements, [e.get_label() for e in legend_elements],
                    loc='center', ncol=len(legend_elements),
                    frameon=True, fancybox=True, shadow=True,
                    fontsize=legend_fontsize)
    
    # Set legend frame properties
    frame = legend.get_frame()
    frame.set_facecolor('black' if dark_mode else 'white')
    frame.set_edgecolor('white' if dark_mode else 'black')
    frame.set_linewidth(2)

    # Set text color
    for text in legend.get_texts():
        text.set_color('white' if dark_mode else 'black')

    legend_ax.add_artist(legend)
    
    
    if not show_legend:
        legend.remove()

    # plt.tight_layout()
    plt.show()

def _interpolate_color(color1: str, color2: str, t: float) -> str:
    """Helper function to interpolate between two colors"""
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*[int(x) for x in rgb])
    
    c1 = hex_to_rgb(color1)
    c2 = hex_to_rgb(color2)
    
    rgb = tuple(int(c1[i] * (1-t) + c2[i] * t) for i in range(3))
    return rgb_to_hex(rgb)








def plot_nn_g_function(activation_function, title="", s_range=(0, 1), phi_range=(-1, 1), i_b=1.9524, resolution=200, dpi=100):
    """
    This function is not critical to anything but is useful for visualising the g function of the SOEN model.
    Its used in one or 2 of the notebooks.
    """

    # from model.soen_model import SOENModel, SOENConfig
    config = SOENConfig(
        activation_function=activation_function,
        clip_phi=False, # if we want to see the full periodicity of the activation function
        clip_state=True,
    )

    model = SOENModel(config)
    model.i_b = i_b
    
    s_values = np.linspace(s_range[0], s_range[1], resolution)
    phi_values = np.linspace(phi_range[0], phi_range[1], resolution)
    s_grid, phi_grid = np.meshgrid(s_values, phi_values)
    
    s_tensor = torch.FloatTensor(s_grid.flatten())
    phi_tensor = torch.FloatTensor(phi_grid.flatten())
    
    with torch.no_grad():
        g_output = model.g(phi_tensor, s_tensor)
    
    g_output = g_output.detach().numpy().reshape(s_grid.shape)
    
    plt.figure(figsize=(10, 8), dpi=dpi)
    plt.contourf(s_grid, phi_grid, g_output, cmap='coolwarm', levels=100)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)  
    cbar.set_label('$g(\phi,s)$', fontsize=25)  

    plt.xlabel('$s$', fontsize=25)  
    plt.ylabel('$\phi$', fontsize=25)  
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.title(f'{title}', fontsize=25)

    plt.tight_layout()
    plt.show()


def plot_batch_state_evolution(state_evolution, num_input, num_hidden, num_output,num_to_plot=1,dpi=200):
    time_steps, batch_size, num_total = state_evolution.shape
    num_to_plot = min(num_to_plot, batch_size)
    
    if num_to_plot == 1:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5),dpi=dpi)
        axs = [axs]  # make it a list to maintain consistent indexing
    else:
        fig, axs = plt.subplots(num_to_plot, 3, figsize=(18, 5*num_to_plot))
    
    for i in range(num_to_plot):
        # input states
        axs[i][0].plot(state_evolution[:, i, :num_input].cpu().numpy())
        axs[i][0].set_title(f"Sample {i+1}: Input States")
        axs[i][0].set_xlabel("Time Step")
        axs[i][0].set_ylabel("State Value")
        
        # hidden states
        axs[i][1].plot(state_evolution[:, i, num_input:num_input+num_hidden].cpu().numpy())
        axs[i][1].set_title(f"Sample {i+1}: Hidden States")
        axs[i][1].set_xlabel("Time Step")
        axs[i][1].set_ylabel("State Value")
        
        # output states
        axs[i][2].plot(state_evolution[:, i, -num_output:].cpu().numpy())
        axs[i][2].set_title(f"Sample {i+1}: Output States")
        axs[i][2].set_xlabel("Time Step")
        axs[i][2].set_ylabel("State Value")

    plt.tight_layout()
    plt.show()




def plot_state_evolution_for_specific_g(model,initial_state,input_values,source_function="tanh",title = "",batch_size = 1):
    # set the activation function
    model.config.activation_function = source_function
    if source_function=="NN_dendrite":
        model._initialise_nn_dendrite()
    output = model(input_values, initial_state=initial_state)
    state_evolution = model.get_state_evolution().detach()
    print(f"g = {source_function}")
    plot_batch_state_evolution(state_evolution, model.config.num_input, model.config.num_hidden, model.config.num_output, num_to_plot=batch_size)
  




#########################################################
#   The following functions SAVE and LOAD SOENModels    #
#########################################################



def save_soen_model(model, filepath):
    """
    Save the SOEN model, including its state dict and configuration.
    """
    saved_data = {
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__,
        'model_class': model.__class__.__name__
    }
    
    print("Saving model with the following keys:", saved_data.keys())  # Debug print
    torch.save(saved_data, filepath)
    print(f"Model saved to {filepath}")

def load_soen_model(filepath,model_class = None):
    
    """
    Load a SOEN model, including its state dict and configuration.
    """
    saved_data = torch.load(filepath)

    # print("Loaded data with the following keys:", saved_data.keys())  # Debug print

    if 'config' not in saved_data:
        raise ValueError("The saved model file does not contain configuration information.")

    
    saved_data = torch.load(filepath)
    model_state_dict = saved_data['model_state_dict']
    model_config = saved_data['config']
    model_config = SOENConfig(**saved_data['config'])
    
    if model_class==None:
        raise TypeError("Remember to pass in the model class as well as the filepath. For example, SOENModel")
    # atm it will only ever be SOENModel but perhaps we might want to create another variant to hace multiple layers
    # or the full dendritic arbor called SOENModelFullNeurons or something similar
    model = model_class(model_config)

    # and then load in all the weights and params
    model.load_state_dict(saved_data['model_state_dict'])


    # print(f"Model loaded from {filepath}")
    return model





#