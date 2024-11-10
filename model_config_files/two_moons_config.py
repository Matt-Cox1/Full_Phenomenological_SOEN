# File: Application/model_config_files/two_moons_config.py


from dataclasses import dataclass, field
from typing import List, Literal
import torch

@dataclass
class TwoMoonsConfig:
    # Network structure
    num_input: int = 2
    num_hidden: int = 10
    num_output: int = 2
    input_type: str = "state"
    is_input_time_varying: bool = False
    input_signal_to_output_nodes: bool = False # should always be false unless we want to clamp the output nodes instead (experimental)

    # Simulation parameters
    dt: float = 0.05
    max_iter: int = 10
    tol: float = 1e-6
    run_to_equilibrium: bool = False  
    track_state_evolution: bool = True
    track_energy: bool = False
    device: torch.device = torch.device("cpu")

    # Weight initialization method
    weight_init_method: Literal["normal","glorot"] = "glorot"
    # Connection mask distribution
    mask_distribution: Literal["bernoulli","power_law"] = 'bernoulli' 


    init_scale: float = 0.1
    enforce_symmetric_weights: bool = True
    bias_flux_offsets: bool = True

    # Node parameters
    gamma_mean: float = 1.0
    gamma_std: float = 0.5
    tau_mean: float = 1.0
    tau_std: float = 0.5

    # Connection probabilities
    p_input_hidden: float = 1
    p_hidden_output: float = 1
    p_input_input: float = 0.0
    p_hidden_hidden: float = 1
    p_output_output: float = 0.0
    allow_output_to_hidden_feedback: bool = True
    allow_hidden_to_input_feedback: bool = True
    allow_skip_connections: bool = False
    allow_self_connections: bool = False
    p_skip_connections: float = 1
    

    # Noise parameters
    train_noise_std: float = 0.05
    test_noise_std: float = 0.05

    enforce_non_negativity_in_gamma: bool = True
    enforce_non_negativity_in_tau: bool = True

    # Learning parameters
    learnable_params: List[str] = field(default_factory=lambda: ["J", "gamma", "tau", "flux_offset"])


    # Activation function parameters
    clip_phi: bool = True
    clip_state: bool = True 


    activation_function: Literal["tanh_1d","tanh_2d" "NN_dendrite","relu_1d","relu_2d","gaussian_mixture","sigmoid_mixture"] = "gaussian_mixture"
    nn_dendrite_model_path: str = "/experiments/source_functions/trained_models/64_hidden_units/64_trained_NN_dendrite.pth" # replace with filepath if need be

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
    
    
    
    
    



