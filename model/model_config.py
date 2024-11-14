# FILENAME: model_config.py
from dataclasses import dataclass, field
from typing import Literal
from typing import List
import torch


"""
This script provides the backbone for the network construction, enabling simulations settings 
to be set, network architecture to be chosen, and experimental features toggled on and off.

- The use of the dataclass decorator allows for the creation of a class without the normal __init__, __repr__, etc methods.
"""

@dataclass
class SOENConfig:
    # Network structure
    num_input: int = 1
    num_hidden: List[int] = field(default_factory=lambda: [1])  # List of hidden layer sizes
    num_output: int = 1
    input_type: str = "state"  # 'state' or 'flux'
    is_input_time_varying: bool = False
    input_signal_to_output_nodes: bool = False # should always be false unless we want to feed the input into the output nodes instead of the input nodes

    # Simulation parameters
    dt: float = 0.05
    max_iter: int = 5
    tol: float = 1e-5
    run_to_equilibrium: bool = False  
    track_state_evolution: bool = False
    track_energy: bool = False
    device: torch.device = torch.device("cuda")# the method of device allocation could be improved

    # Initialisation parameters
    # Weight initialisation method
    weight_init_method: Literal["normal","glorot"] = "normal"
    # Connection mask distribution
    mask_distribution: Literal["bernoulli","power_law"] = 'bernoulli' 



    init_scale: float = 0.1
    enforce_symmetric_weights: bool = False
    bias_flux_offsets: bool = False

    # Node parameters
    gamma_mean: float = 1.0
    gamma_std: float = 0.5
    tau_mean: float = 1.0
    tau_std: float = 0.5

  

    enforce_non_negativity_in_gamma: bool = True
    enforce_non_negativity_in_tau: bool = True

    # Connection probabilities
    p_input_input: float = 1.0
    p_output_output: float = 1.0

    p_input_hidden: float =1.0 # equal to the prob of connecting a hidden to input node if allow_hidden_to_input_feedback=True
    p_hidden_self: float = 1.0 # prob of a hidden node connecting to itself (like in a RNN)
    p_hidden_hidden: float = 1.0 # prob of connecting to a different hidden node in the same layer
    
    # p_inter_hidden: List[int] = field(default_factory=lambda:[1.0]) #when there is more than one hidden layer, this defines forward connection probabilities from successive hidden layers to the next one
    p_inter_hidden: float = 1.0 #when there is more than one hidden layer, this defines forward connection probabilities from successive hidden layers to the next one
    # note that when there's only one hidden layer, this is unused



    allow_self_connections: bool = False
    allow_output_to_hidden_feedback: bool = False
    allow_hidden_to_input_feedback: bool = False
    # Skip connection parameters
    allow_skip_connections: bool = True
    p_skip_connections: float = 1.0  # Probability of skip connections (input to output in this case because we only have 1 hidden layer)

    # Flux noise parameters
    train_noise_std: float = 0.00
    test_noise_std: float = 0.00

    # Learning parameters
    # TO DO: Add bias current as a learnable parameter
    learnable_params: List[str] = field(default_factory=lambda: ["J", "gamma", "tau", "flux_offset"])


    # Parameter constraints
    clip_phi: bool = True # clips between -0.5 and 0.5
    clip_state: bool = True # clips between 0 and 1


    activation_function: Literal["tanh_1d","tanh_2d" "NN_dendrite","relu_1d","relu_2d","gaussian_mixture","sigmoid_mixture"] = "gaussian_mixture"
    nn_dendrite_model_path: str = "experiments/source_functions/trained_models/64_hidden_units/64_trained_NN_dendrite.pth"
    

    # The 2 proposed in the report are the energy_soen_local and energy_soen_global
    energy_function: Literal[
            "energy_soen_local",
            "energy_soen_global",
            "energy_soen_2",
            "energy_soen_3",
            "energy_soen_4",
            "energy_sam",
            "energy_hopfield",
            "energy_simple",
            "energy_kinetic",
        ] ="energy_soen_local"
        
        
        
    
