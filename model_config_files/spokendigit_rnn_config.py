# FILENAME: Application/model_config_files/spokendigit_rnn_config.py

#todos
#add linear offset to spoken digit data generation to mask weak signals (in log case?)
#think about how to use dendrites to fake a sigmoid function, or alternatively just train a network to do that and reuse that block! (unclamped case)
#fix MA issues at beginning of audio files
#still need to try unclamped phi/state to see if that makes things a lot worse

#

from dataclasses import dataclass, field
from typing import List, Literal
import torch

@dataclass
class SpokenDigitRNNConfig:
    # Network structure
    num_input: int = 32
    num_hidden: List[int] = field(default_factory=lambda: [64])  # List of hidden layer sizes
    num_output: int = 10
    input_type: str = "state"
    is_input_time_varying: bool = True
    input_signal_to_output_nodes: bool = False # should always be false unless we want to clamp the output nodes instead (experimental)

    batch_size: int = 64
    # Simulation parameters
    dt: float = .025
    max_iter: int = 40
    tol: float = 1e-6
    run_to_equilibrium: bool = False  
    track_state_evolution: bool = True
    track_energy: bool = False
    device: torch.device = torch.device("cuda") # here you can change the device to cuda, mps or cpu

    # Weight initialisation method
    weight_init_method: Literal["normal","glorot"] = "glorot"
    z_weight_init_method: Literal["normal","glorot"] = "normal"
    # Connection mask distribution
    mask_distribution: Literal["bernoulli","power_law_except_self"] = 'bernoulli' 

    # only_self_connections_in_hidden: bool = True # only allow self connections in the hidden layer
    init_scale: float = 0.1
    z_init_scale: float = 0.3
    enforce_symmetric_weights: bool = False
    bias_flux_offsets: bool = True

    # Node parameters
    gamma_mean: float = 1.0
    gamma_std: float = 0.8
    tau_mean: float = 1.0
    tau_std: float = 0.8
    layer_gamma_mean: float = 1.0
    layer_gamma_std: float = 0.8
    layer_tau_mean: float = 1.0
    layer_tau_std: float = 0.8

    # Connection probabilities
    p_input_hidden: float = 1.0
    p_hidden_output: float = 1.0
    p_input_input: float = 0.0
    p_hidden_self: float = 1.0
    p_hidden_hidden: float = 0.0
    p_inter_hidden: float = 1.0
    p_output_output: float = 0.0
    allow_output_to_hidden_feedback: bool = False
    allow_hidden_to_input_feedback: bool = False
    allow_skip_connections: bool = False
    p_skip_connections: float = 0.05
    allow_self_connections: bool = True

    # Noise parameters
    train_noise_std: float = 0.0
    test_noise_std: float = 0.0

    enforce_layer_uniformity_in_gamma: bool = False
    enforce_layer_uniformity_in_tau: bool = False

    enforce_Z_disabled: bool = False



    enforce_non_negativity_in_gamma: bool = True
    enforce_non_negativity_in_tau: bool = True

    # Learning parameters
    learnable_params: List[str] = field(default_factory=lambda: ["J", "JZ","gamma", "tau", "flux_offset", "flux_offset_Z"])
    # Activation function parameters
    clip_phi: bool = True
    clip_state: bool = True 


    activation_function: Literal["tanh_1d","tanh_2d" "NN_dendrite","relu_1d","relu_2d","gaussian_mixture","sigmoid_mixture"] = "NN_dendrite"
    nn_dendrite_model_path: str = "experiments/source_functions/trained_models/64_hidden_units/64_trained_NN_dendrite.pth" # replace with filepath if need be
    

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
    ] ="energy_simple"
    

    def __post_init__(self):
        self.num_hidden_layers = len(self.num_hidden)



    
    
    
    



