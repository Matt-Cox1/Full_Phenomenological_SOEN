# FILENAME: utils/network_analysis.py

import torch
from typing import Dict, Any
import shutil

def format_number(num: float) -> str:
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def analyse_network_structure(model: Any) -> Dict[str, int]:
    return {
        "input_nodes": model.num_input,
        "hidden_nodes": model.total_hidden,
        "hidden_layers": model.num_hidden_layers,
        "hidden_layer_sizes": model.num_hidden,
        "output_nodes": model.num_output,
        "total_nodes": model.num_total
    }

def analyse_parameters(model: Any) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_mask = torch.sum(model.mask).item()
    non_zero_weights = torch.sum(model.J * model.mask != 0).item()

    params_info = {}
    for name, param in model.named_parameters():
        if name == 'J':
            params_info[name] = {
                "shape": tuple(param.shape),
                "non_zero_elements": format_number(non_zero_weights)
            }
        else:
            params_info[name] = {
                "shape": tuple(param.shape),
                "elements": format_number(param.numel())
            }

    return {
        "params_info": params_info,
        "total_params": total_params,
        "non_zero_mask": non_zero_mask,
        "non_zero_weights": non_zero_weights
    }

def analyse_weight_matrix(model: Any) -> Dict[str, Any]:
    non_zero_weights = torch.sum(model.J * model.mask != 0).item()
    sparsity = 1 - (non_zero_weights / model.J.numel())
    return {
        "shape": tuple(model.J.shape),
        "non_zero": non_zero_weights,  # Store as a number, not a formatted string
        "sparsity": sparsity
    }


def analyse_overall_statistics(model: Any) -> Dict[str, Any]:
    total_possible = model.num_total * model.num_total
    non_zero_weights = torch.sum(model.J * model.mask != 0).item()
    sparsity = 1 - (non_zero_weights / model.J.numel())
    return {
        "total_possible": total_possible,
        "actual_connections": non_zero_weights,
        "sparsity": sparsity
    }

def analyse_connection_distribution(model: Any) -> Dict[str, Any]:
    connections_per_node = torch.sum(model.mask, dim=1)
    return {
        "mean_connections": connections_per_node.float().mean().item(),
        "std_connections": connections_per_node.float().std().item()
    }

def analyse_parameter_statistics(model: Any) -> Dict[str, Dict[str, float]]:
    param_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item()
            }
    return param_stats


def analyse_node_connections(model: Any) -> Dict[str, Any]:
    connections_per_node = torch.sum(model.mask, dim=1)
    
    # Get layer start indices from the model
    layer_starts = model.layer_starts
    
    # Analyze input layer connections
    input_connections = analyse_node_connections_layer(
        connections_per_node[:model.num_input]
    )
    
    # Analyze each hidden layer separately
    hidden_layer_connections = []
    for layer in range(model.num_hidden_layers):
        start_idx = layer_starts[layer]
        end_idx = layer_starts[layer + 1]
        layer_connections = analyse_node_connections_layer(
            connections_per_node[start_idx:end_idx]
        )
        hidden_layer_connections.append(layer_connections)
    
    # Analyze output layer connections
    output_connections = analyse_node_connections_layer(
        connections_per_node[-model.num_output:]
    )
    
    return {
        "input_layer": input_connections,
        "hidden_layers": hidden_layer_connections,
        "output_layer": output_connections
    }

def analyse_node_connections_layer(connections: torch.Tensor) -> Dict[str, Any]:
    if connections.numel() == 0:
        return {"empty": True, "message": "No connections found for this layer."}
    
    connection_counts = torch.bincount(connections.long())
    max_connections = connection_counts.numel() - 1
    
    if max_connections == 0:
        return {"empty": True, "message": "All nodes in this layer have 0 connections."}
    
    max_bar_height = connection_counts.max().item()
    distribution = {}
    for i in range(max_connections + 1):
        if connection_counts[i] > 0:
            bar_height = int(connection_counts[i] / max_bar_height * 50)
            distribution[i] = {
                "count": connection_counts[i].item(),
                "bar": '█' * bar_height
            }
    return {"empty": False, "distribution": distribution}


def print_section(title: str, content: str) -> None:
    # Get the terminal width, or use a default of 80 if not available
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    width = max(40, min(width, 80))  # Ensure width is between 40 and 80

    def print_border():
        print("─" * width)

    # Print top border and title
    print_border()
    print(f"{title.upper():^{width}}")
    print_border()

    # Print content
    for line in content.split('\n'):
        print(f"{line:<{width}}")

    # Print bottom border
    print_border()
    print() 



    
def print_analysis(analysis: Dict[str, Any], verbosity_level: int = 3) -> None:
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    width = max(40, min(width, 80))

    print("─" * width)
    print(f"{'SOEN Network Analysis':^{width}}")
    print("─" * width)
    print()

    if verbosity_level >= 1 and "network_structure" in analysis:
        content = [
            f"{'Input Nodes':15} {analysis['network_structure']['input_nodes']:,}",
            f"{'Hidden Layers':15} {analysis['network_structure']['hidden_layers']:,}",
            f"{'Total Hidden':15} {analysis['network_structure']['hidden_nodes']:,}"
        ]
        # Add individual layer sizes
        for i, size in enumerate(analysis['network_structure']['hidden_layer_sizes']):
            content.append(f"{'Layer ' + str(i+1) + ' Size':15} {size:,}")
        content.append(f"{'Output Nodes':15} {analysis['network_structure']['output_nodes']:,}")
        content.append(f"{'Total Nodes':15} {analysis['network_structure']['total_nodes']:,}")
        
        print_section("Network Structure", "\n".join(content))

    if verbosity_level >= 2 and "parameters" in analysis:
        content = ""
        for name, info in analysis["parameters"]["params_info"].items():
            content += f"{name:12} Shape {str(info['shape']):15} "
            content += f"{'Non-zero elements' if name == 'J' else 'Elements':20} {info['non_zero_elements' if name == 'J' else 'elements']:>10}\n"
        content += f"\nTotal Parameters:        {analysis['parameters']['total_params']:,}\n"
        content += f"Non-zero Mask:           {analysis['parameters']['non_zero_mask']:,}\n"
        content += f"Non-zero Weights:        {analysis['parameters']['non_zero_weights']:,}"
        print_section("Parameters", content)

    if verbosity_level >= 3:
        if "weight_matrix" in analysis:
            wm = analysis["weight_matrix"]
            content = f"Shape:                {wm['shape']}\n"
            content += f"Non-zero Elements:    {format_number(wm['non_zero'])}\n"
            content += f"Sparsity:             {wm['sparsity']*100:.2f}%"
            print_section("Weight Matrix Analysis", content)

        if "overall_statistics" in analysis:
            os = analysis["overall_statistics"]
            content = f"Total Possible Connections:   {os['total_possible']:,}\n"
            content += f"Actual Connections:           {os['actual_connections']:,}\n"
            content += f"Sparsity:                     {os['sparsity']*100:.2f}%"
            print_section("Overall Statistics", content)

            
        # node-wise connection histograms
        for layer_name in ["input", "hidden", "output"]:
            connections = analysis[f"{layer_name}_node_connections"]
            if not connections["empty"]:
                content = f"Connection distribution for {layer_name} nodes:\n\n"
                content += "Connections | Count | Distribution\n"
                content += "-" * (width - 2) + "\n"
                for connections, data in connections["distribution"].items():
                    content += f"{connections:11d} | {data['count']:5d} | {data['bar']}\n"
            else:
                content = connections["message"]
            print_section(f"{layer_name.capitalize()} Node Connections", content)




def format_network_analysis(analysis: Dict[str, Any]) -> str:
    def format_section(title, content):
        section = f"\n{'─' * 80}\n"
        section += f"  {title.upper()}\n"
        section += f"{'─' * 80}\n"
        section += content + "\n"
        return section

    def format_distribution(distribution_data):
        if distribution_data.get("empty", False):
            return distribution_data.get("message", "No data available.")
        
        content = "Connections | Count | Distribution\n"
        content += "───────────┼───────┼────────────────────────────────────────────\n"
        for connections, data in distribution_data["distribution"].items():
            bar = data["bar"]
            count = data["count"]
            content += f"{connections:11d} | {count:5d} | {bar}\n"
        return content

    def format_param_stats(param_stats):
        content = "Parameter   |   Mean   |   Std Dev\n"
        content += "────────────┼──────────┼───────────\n"
        for param, stats in param_stats.items():
            content += f"{param:11} | {stats['mean']:8.4f} | {stats['std']:9.4f}\n"
        return content

    formatted_text = "SOEN Network Analysis\n"
    formatted_text += "=" * 80 + "\n\n"

    if "network_structure" in analysis:
        content = []
        for k, v in analysis["network_structure"].items():
            if k == "hidden_layer_sizes":
                content.append(f"{'Hidden Layer Sizes':15} {str(v)}")
            else:
                content.append(f"{k.replace('_', ' ').title():15} {v:,}")
        formatted_text += format_section("Network Structure", "\n".join(content))

    if "parameters" in analysis:
        content = ""
        for name, info in analysis["parameters"]["params_info"].items():
            content += f"{name:12} Shape {str(info['shape']):15} "
            content += f"{'Non-zero elements' if name == 'J' else 'Elements':20} {info['non_zero_elements' if name == 'J' else 'elements']}\n"
        content += f"\nTotal Parameters:        {analysis['parameters']['total_params']}\n"
        content += f"Non-zero Mask:           {analysis['parameters']['non_zero_mask']}\n"
        content += f"Non-zero Weights:        {analysis['parameters']['non_zero_weights']}"
        formatted_text += format_section("Parameters", content)

    if "weight_matrix" in analysis:
        wm = analysis["weight_matrix"]
        content = f"Shape:                {wm['shape']}\n"
        content += f"Non-zero Elements:    {wm['non_zero']}\n"
        content += f"Sparsity:             {wm['sparsity']*100:.2f}%"
        formatted_text += format_section("Weight Matrix Analysis", content)

    if "overall_statistics" in analysis:
        os = analysis["overall_statistics"]
        content = f"Total Possible Connections:   {os['total_possible']:,}\n"
        content += f"Actual Connections:           {os['actual_connections']:,}\n"
        content += f"Sparsity:                     {os['sparsity']*100:.2f}%"
        formatted_text += format_section("Overall Statistics", content)

    if "parameter_statistics" in analysis:
        formatted_text += format_section("Parameter Statistics", format_param_stats(analysis["parameter_statistics"]))

    if "input_node_connections" in analysis:
        formatted_text += format_section("Input Node Connectivity Distribution", 
                                         format_distribution(analysis["input_node_connections"]))

    if "hidden_node_connections" in analysis:
        # Handle multiple hidden layers
        for i, layer_data in enumerate(analysis["hidden_node_connections"]):
            formatted_text += format_section(f"Hidden Layer {i+1} Connectivity Distribution", 
                                          format_distribution(layer_data))

    if "output_node_connections" in analysis:
        formatted_text += format_section("Output Node Connectivity Distribution", 
                                         format_distribution(analysis["output_node_connections"]))

    return formatted_text

def analyse_network(model: Any, verbosity_level: int = 3) -> Dict[str, Any]:
    analysis = {}
    
    if verbosity_level >= 1:
        analysis["network_structure"] = analyse_network_structure(model)
    
    if verbosity_level >= 2:
        analysis["parameters"] = analyse_parameters(model)
    
    if verbosity_level >= 3:
        analysis["weight_matrix"] = analyse_weight_matrix(model)
        analysis["overall_statistics"] = analyse_overall_statistics(model)
        analysis["connection_distribution"] = analyse_connection_distribution(model)
        
        # Pass the model directly to analyse_node_connections
        node_connections = analyse_node_connections(model)
        analysis["input_node_connections"] = node_connections["input_layer"]
        analysis["hidden_node_connections"] = node_connections["hidden_layers"]
        analysis["output_node_connections"] = node_connections["output_layer"]
        
        analysis["parameter_statistics"] = analyse_parameter_statistics(model)
    
    # Print the formatted analysis
    print(format_network_analysis(analysis))
    
    return analysis