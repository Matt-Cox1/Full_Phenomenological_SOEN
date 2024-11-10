# FILENAME: model/soen_model.py


import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model.model_config import SOENConfig
from utils.soen_model_utils import *
from utils.activation_functions import *
from utils.network_analysis import analyse_network, print_analysis
from utils.soen_model_helpers import *
from utils.energy_functions import *



from scipy.integrate import quad
from scipy.integrate import IntegrationWarning
import warnings
import random
import numpy as np
import math
import logging
from copy import deepcopy # to make a separate copy of the config file so we dont accidentally overwrite it. 

torch.manual_seed(42)
np.random.seed(42)
# note: I specifically did not set random's seed. The random module is used to temporarily un-set the torch seed when initialising random model parameters



class SOENModel(nn.Module):
    def __init__(self, config: SOENConfig):
        super(SOENModel, self).__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__) # set the logger to the class name, so we can see which class is logging what
        self.logger.setLevel(logging.DEBUG) # set the logger to the debug level so we can see everything


        # to avoid writing over config variables, we make a deep copy of the config file
        self.config = deepcopy(config)
        self.device = torch.device(config.device)
        self._initialise_network_parameters() # things like: dt, max_iter, tol, etc
        self._initialise_network_structure() # num_input, num_hidden, num_output, etc
        self._initialise_model_parameters() # J, gamma, tau, etc
        self.to(self.device) 
        
        
        
        # list the activation functions that can be used, as found in the activation_functions.py file
        self.activation_functions = {
            "tanh_2d": tanh_2d,
            "relu_2d": relu_2d,
            "gaussian_mixture": gaussian_mixture,
            "sigmoid_mixture": sigmoid_mixture,
            "NN_dendrite": None,
            "tanh_1d": tanh_1d,
            "relu_1d": relu_1d,}

        self._initialise_nn_dendrite() # this makes the nn_dendrite a callable function not to be learned
        
        
        
        if self.bias_flux_offsets:
            self.set_initial_flux_offset()
        
        # For every source function, we want to make it periodic
        for key in self.activation_functions:
            if self.activation_functions[key] is not None:
                self.activation_functions[key] = make_periodic(self.activation_functions[key])
        
        
        self.state_evolution = None
        self.energy_evolution = None if not self.track_energy else []
    
    def _initialise_nn_dendrite(self):
        # if the activation function as specified in the config file is the neural network
        # dendrite, then we need to initialise the neural network dendrite wrapper
        if self.config.activation_function == "NN_dendrite":
            self.nn_dendrite_wrapper = NNDendriteWrapper(
                self.config.nn_dendrite_model_path, 
                self.i_b
            ).to(self.device)
            self.activation_functions["NN_dendrite"] = self.nn_dendrite_wrapper

    def g(self, phi, s):
        return self.activation_functions[self.config.activation_function](phi, s)
    

    def set_activation_function(self, activation_function):
        self.config.activation_function = activation_function
        if activation_function == "NN_dendrite":
            self._initialise_nn_dendrite()

            
    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        super().to(device)
        if hasattr(self, 'nn_dendrite_wrapper') and self.nn_dendrite_wrapper is not None:
            self.nn_dendrite_wrapper = self.nn_dendrite_wrapper.to(device)
        self.J = self.J.to(device)
        self.gamma = self.gamma.to(device)
        self.tau = self.tau.to(device)
        self.flux_offset = self.flux_offset.to(device)
        self.external_flux = self.external_flux.to(device)
        self.mask = self.mask.to(device)
        self.i_b = self.i_b.to(device)  
        return self

    def _initialise_network_structure(self):
        self.num_input = self.config.num_input
        self.num_hidden = self.config.num_hidden
        self.num_output = self.config.num_output
        self.num_total = self.num_input + self.num_hidden + self.num_output
        self.input_type = self.config.input_type

    def _initialise_network_parameters(self):
        self.bias_flux_offsets = self.config.bias_flux_offsets
        self.track_state_evolution = self.config.track_state_evolution
        self.track_energy = self.config.track_energy
        self.energy_function = globals()[self.config.energy_function]
        self.dt = self.config.dt
        self.max_iter = self.config.max_iter
        self.tol = self.config.tol
        self.train_noise_std = self.config.train_noise_std
        self.test_noise_std = self.config.test_noise_std
        self.clip_phi = self.config.clip_phi
        self.clip_state = self.config.clip_state
        self.enforce_non_negativity_in_gamma = self.config.enforce_non_negativity_in_gamma
        self.enforce_non_negativity_in_tau = self.config.enforce_non_negativity_in_tau
        
        self.allow_self_connections = self.config.allow_self_connections
        self.allow_output_to_hidden_feedback = self.config.allow_output_to_hidden_feedback
        self.allow_hidden_to_input_feedback = self.config.allow_hidden_to_input_feedback
        self.enforce_symmetric_weights = self.config.enforce_symmetric_weights

    def _initialise_model_parameters(self):
        self._create_connection_mask()

        # Generate a random seed using Python's random module
        random_seed = random.randint(0, 2**32 - 1)

        # Set PyTorch's seed for parameter initialisation
        torch.manual_seed(random_seed)

        # Initialise parameters
        if self.config.weight_init_method == "normal":
            self.J = nn.Parameter(torch.randn(self.num_total, self.num_total) * self.config.init_scale)
        elif self.config.weight_init_method == "glorot":
            self.J = self._initialise_glorot_weights()
        else:
            raise ValueError(f"Unknown weight initialisation method: {self.config.weight_init_method}")

        self.gamma = nn.Parameter(torch.normal(mean=self.config.gamma_mean, std=self.config.gamma_std, size=(self.num_total,)).abs())
        self.tau = nn.Parameter(torch.normal(mean=self.config.tau_mean, std=self.config.tau_std, size=(self.num_total,)).abs())

        # Reset PyTorch's seed back to 42 for consistency with the rest of the code
        torch.manual_seed(42)

        self.flux_offset = nn.Parameter(torch.zeros(self.num_total))
        self.i_b = torch.tensor(1.7)

        self._apply_mask_to_parameters()
        self._set_parameter_gradients()
        




    
    def _initialise_glorot_weights(self):
        """
        Initialise weights using Glorot (Xavier) initialisation.
        This method aims to maintain the variance of activations and gradients across layers, and has been adapted for SOENs
        to act node-wise rather than layers, but it follows the same principles.
        """
        # how many connections are coming into each node and how many are going out (setting the min to be 1 to avoid a division by zero)
        incoming_connections = torch.clamp(self.get_incoming_connections(), min=1)
        outgoing_connections = torch.clamp(torch.sum(self.mask, dim=1), min=1)
        
        # Calculate the limit for uniform distribution (we'll call it `a`)
        a = torch.sqrt(6.0 / (incoming_connections + outgoing_connections))
        
        # for each weight we then samples from a uniform distribution between -a and a
        weights = torch.zeros(self.num_total, self.num_total)
        for i in range(self.num_total):
            for j in range(self.num_total):
                if self.mask[i, j]:
                    weights[i, j] = torch.empty(1).uniform_(-a[i], a[i])
        
        return nn.Parameter(weights)
    


    def _create_connection_mask(self):
        """
        Create a mask for connections in the SOEN network using either power law or Bernoulli distribution.
        """
        mask = torch.zeros(self.num_total, self.num_total)
        
        def create_mask(rows, cols, density, distribution='bernoulli'):
            if distribution == 'power_law':
                alpha = 1.1 # this is a steepness of the power law distribution
                return self._create_power_law_mask(rows, cols, density,alpha)
            else:  # default to Bernoulli
                return torch.bernoulli(torch.full((rows, cols), density, dtype=torch.float32))


        # Input to Hidden connections
        input_hidden_mask = create_mask(self.num_hidden, self.num_input, self.config.p_input_hidden, self.config.mask_distribution)
        mask[self.num_input:self.num_input+self.num_hidden, :self.num_input] = input_hidden_mask
        
        # Ensure bidirectional consistency for input-hidden connections
        if self.config.allow_hidden_to_input_feedback:
            mask[:self.num_input, self.num_input:self.num_input+self.num_hidden] = input_hidden_mask.t()
        
        # Hidden to Output connections
        if self.num_output > 0:
            hidden_output_mask = create_mask(self.num_output, self.num_hidden, self.config.p_hidden_output, self.config.mask_distribution)
            mask[self.num_input+self.num_hidden:, self.num_input:self.num_input+self.num_hidden] = hidden_output_mask
            
            if self.config.allow_output_to_hidden_feedback:
                mask[self.num_input:self.num_input+self.num_hidden, self.num_input+self.num_hidden:] = hidden_output_mask.t()
        
        # Input to Input connections
        if self.config.p_input_input > 0:
            input_input_mask = create_mask(self.num_input, self.num_input, self.config.p_input_input, self.config.mask_distribution)
            mask[:self.num_input, :self.num_input] = input_input_mask
        
        # Hidden to Hidden connections
        if self.config.p_hidden_hidden > 0:
            hidden_hidden_mask = create_mask(self.num_hidden, self.num_hidden, self.config.p_hidden_hidden, self.config.mask_distribution)
            mask[self.num_input:self.num_input+self.num_hidden, self.num_input:self.num_input+self.num_hidden] = hidden_hidden_mask
        
        # Output to Output connections
        if self.config.p_output_output > 0 and self.num_output > 0:
            output_output_mask = create_mask(self.num_output, self.num_output, self.config.p_output_output, self.config.mask_distribution)
            mask[self.num_input+self.num_hidden:, self.num_input+self.num_hidden:] = output_output_mask
        
        # Skip connections (Input to Output)
        if self.config.allow_skip_connections and self.num_output > 0:
            skip_mask = create_mask(self.num_output, self.num_input, self.config.p_skip_connections, self.config.mask_distribution)
            mask[self.num_input+self.num_hidden:, :self.num_input] = skip_mask
            mask[:self.num_input, self.num_input+self.num_hidden:] = skip_mask.t()
        
        # Remove self-connections if not allowed
        if not self.config.allow_self_connections:
            mask.diagonal().zero_()
        
        # Enforce symmetry if required
        if self.config.enforce_symmetric_weights:
            mask = torch.max(mask, mask.t())
        
        self.register_buffer('mask', mask.t())

    # def _create_power_law_mask(self, rows, cols, density):
    #     """Generate a power law distribution mask."""
    #     mask = torch.zeros(rows, cols)
        
    #     for i in range(rows):
    #         x = torch.arange(1, cols + 1, dtype=torch.float32)
    #         prob = 1 / x.pow(2)  # Using 1/x^2 distribution
    #         prob /= prob.sum()
    #         num_connections = int(cols * density)
    #         connected = torch.multinomial(prob, num_connections, replacement=False)
    #         mask[i, connected] = 1
    #     return mask

    def _create_power_law_mask(self, rows, cols, density, alpha=2.1):
        """Generate a power law distribution mask with many nodes having few connections."""
        mask = torch.zeros(rows, cols)
        total_connections = int(rows * cols * density)
        
        # Generate inverse power law distributed degrees
        x = np.arange(1, rows + 1)
        degrees = x ** (-alpha)
        degrees = degrees / degrees.sum() * total_connections
        degrees = np.round(degrees).astype(int)
        
        # Ensure we have the correct number of total connections
        while degrees.sum() != total_connections:
            if degrees.sum() < total_connections:
                idx = np.random.choice(np.where(degrees < cols)[0])
                degrees[idx] += 1
            else:
                idx = np.random.choice(np.where(degrees > 0)[0])
                degrees[idx] -= 1
        
        for i in range(rows):
            if degrees[i] > 0:
                connected = torch.randperm(cols)[:degrees[i]]
                mask[i, connected] = 1
        
        return mask


    


    def _apply_mask_to_parameters(self):
        self.J.data *= self.mask
        if self.enforce_symmetric_weights:
            self.J.data = (self.J.data + self.J.data.t()) / 2
        self.register_buffer('external_flux', torch.zeros(self.num_total)) # a register buffer is a tensor that is not a parameter (so not learnable) but is part of the model state

    def _set_parameter_gradients(self):
        for param_name in ['J', 'gamma', 'tau', 'flux_offset', 'i_b']:
            param = getattr(self, param_name)
            if param_name in self.config.learnable_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    def enforce_symmetry(self, J=None):
        '''
        This method makes sure that if we want the weights to remain symmetric, they do.
        It is done by making sure the transpose of the weight matrix is the same as the weight matrix,
        and takes the average of the two.
        '''
        if J is None:
            J = self.J.data
        if self.enforce_symmetric_weights:
            J_before = J.clone()
            J = (J + J.t()) / 2 
            J *= self.mask
        return J
        
    def apply_constraints(self, J=None):
        if J is None:
            J = self.J
        with torch.no_grad():
            if self.enforce_non_negativity_in_gamma:
                self.gamma.data = self.gamma.data.abs()
                self.gamma.data = torch.clamp(self.gamma.data, 0.0, 10.0)
            if self.enforce_non_negativity_in_tau:
                self.tau.data = self.tau.data.abs()
                self.tau.data = torch.clamp(self.tau.data, 0.0001, 10.0) # we dont want to divide by zero (in the s/tau calc)
            self.flux_offset.data = torch.clamp(self.flux_offset.data, -0.5, 0.5) # there is no need for the flux offset to be outside of this range because it is periodic
        
        return J

 


    def calculate_energy(self, s, phi):
        return self.energy_function(self, s, phi)
   

    



    def forward(self, x, initial_state=None, return_intermediates=False):
        """
        This is the main function that performs the forward pass of the SOEN model.
        It simulates the dynamics of the network for a given input signal x and initial state vector.
        """

        # place the inputs onto the device
        x = x.to(self.device)
        # if we provided an input state vector then place it on the device
        if initial_state is not None:
            initial_state = initial_state.to(self.device)

        

        self.J.data = self.apply_constraints(self.J.data)
        # enforce symmetry on the weights only if active in the config file
        self.enforce_symmetry()

        batch_size = x.shape[0]
        
        # initialise the state vector
        s = initialise_state(batch_size, self.num_total, initial_state, x.device, self.clip_state)
        
        # Determine which of the nodes to apply the input to (the order of the nodes is input, hidden,output so its just the first nun_input nodes)
        # although for some experiments i have wanted to apply the inputs to the output nodes so i have added a config option for that
        input_nodes = slice(-self.num_output, None) if self.config.input_signal_to_output_nodes else slice(None, self.num_input)  
        
        # Check input shape (just a check to see if the shape of the input is appropriate, depends if the input is meant 
        # to be time varying or not)
        self._check_input_shape(x)

        if not self.config.is_input_time_varying and self.input_type == 'state':
                s[:, input_nodes] = x  # ensure the input states if we're clamping, remain clamped for non-time-varying inputs

        
        # intermediates are the states of the network at each time step, useful to save for some ML algos such as trunctated backprop through time 
        intermediates = [s.clone()] if return_intermediates else None
        
        if self.track_state_evolution:
            self.state_evolution = [s.clone()]
        
        if self.config.track_energy:
            self.energy_evolution = []
        
        # we need to make sure connections dont just start being formed if we're learning J. So multiply J by the binary connection mask
        J_masked = self.J * self.mask
        

        # right, now that everything is set up, we can start the simulation:
        #######################################
        #         MAIN SIMULATION LOOP        #
        #######################################
        for t in range(self.max_iter-1):
            # make a copy of the state vector
            s_old = s.clone()
            
            # Apply input if the input is flux
            external_flux = apply_input(s, x, input_nodes, self.input_type, self.config.is_input_time_varying, t)
            
            # calculate the phi values of every node and over the batch
            phi = calculate_phi(s, J_masked, self.flux_offset, external_flux)

            if self.clip_phi:
                phi = torch.clamp(phi, -0.5, 0.5)
            
            # apply the noise to the flux (phi)values
            phi = apply_noise(phi, self.train_noise_std if self.training else self.test_noise_std, self.training)

            # update the state according to the main ds/dt equation
            s = update_state(s, phi, self.g, self.gamma, self.tau, self.dt, self.clip_state)
            
            if not self.config.is_input_time_varying and self.input_type == 'state':
                s[:, input_nodes] = x  # ensure the input states if we're clamping, remain clamped for non-time-varying inputs

            # Track energy and state evolution
            self._track_evolution(s, phi)
            
            if return_intermediates:
                intermediates.append(s.clone())
            
            if self.config.run_to_equilibrium and check_convergence(s, s_old, self.tol):
                break

        self.enforce_symmetry()
        self.apply_constraints()

        if return_intermediates:
            return s[:, -self.num_output:], intermediates
        else:
            return s[:, -self.num_output:]

    def _check_input_shape(self, x):
        if self.config.is_input_time_varying:
            expected_shape = (self.num_output if self.config.input_signal_to_output_nodes else self.num_input)
            if len(x.shape) != 3 or x.shape[1] != expected_shape:
                raise ValueError(f"Expected time-varying input with shape (batch_size, {expected_shape}, time_steps), "
                                 f"but got shape {x.shape}. "
                                 f"If you intended to use non-time-varying input, set is_input_time_varying=False in the config.")
            
            num_time_steps = x.shape[2]
            expected_time_steps = int(self.max_iter)
            
            if num_time_steps != expected_time_steps:
                print(f"Warning: Input signal length ({num_time_steps}) does not match max_iter ({expected_time_steps}).")
                print(f"Adjusting max_iter to match input signal length.")
                self.max_iter = num_time_steps
        else:
            expected_shape = (self.num_output if self.config.input_signal_to_output_nodes else self.num_input)
            if len(x.shape) != 2 or x.shape[1] != expected_shape:
                raise ValueError(f"Expected non-time-varying input with shape (batch_size, {expected_shape}), "
                                 f"but got shape {x.shape}. "
                                 f"If you intended to use time-varying input, set is_input_time_varying=True in the config.")
 
    def _track_evolution(self, s, phi):
        if self.config.track_energy:
            energy = self.calculate_energy(s, phi)
            self.energy_evolution.append(energy)
            
        if self.track_state_evolution:
            self.state_evolution.append(s.clone())

    def get_energy_evolution(self):
        if self.config.track_energy:
            if self.energy_evolution and len(self.energy_evolution) > 0:
                return torch.stack(self.energy_evolution)
            else:
                return torch.tensor([])  # Return empty tensor if no energy was tracked
        else:
            raise ValueError("Energy evolution tracking is not enabled. Set 'track_energy' to True in the config.")

    def get_state_evolution(self):
        if self.track_state_evolution:
            if self.state_evolution and len(self.state_evolution) > 0:
                return torch.stack(self.state_evolution)
            else:
                return torch.tensor([])  # Return empty tensor if no states were tracked
        else:
            raise ValueError("State evolution tracking is not enabled. Set 'track_state_evolution' to True in the config.")

    def get_incoming_connections(self):
        """
        Returns a tensor containing the number of incoming connections for each node.
        """
        return torch.sum(self.mask, dim=0)

    def analyse_network(self, verbosity_level=3):
        analysis_result = analyse_network(self, verbosity_level)
        # print_analysis(analysis_result)
        return analysis_result

    def find_initial_flux_offset(self, threshold=0.01, phi_start=0, phi_step=0.01, phi_max=0.5, num_s_samples=100):
        """
        Finds a good initial flux offset for all nodes by sweeping through phi values
        until the average g value exceeds a specified threshold.

        Args:
        threshold (float): The threshold for the average g value (default: 0.01)
        phi_start (float): The starting value for phi (default: 0)
        phi_step (float): The step size for incrementing phi (default: 0.01)
        phi_max (float): The maximum value for phi (default: 2.0)
        num_s_samples (int): The number of s values to sample for averaging (default: 100)

        Returns:
        float: The found flux offset value
        """
        s_values = torch.linspace(0, 1, num_s_samples).to(self.device)
        phi = phi_start

        while phi <= phi_max:
            phi_tensor = torch.full((num_s_samples,), phi).to(self.device)
            g_values = self.g(phi_tensor, s_values)
            avg_g = g_values.mean().item()

            if avg_g > threshold:
                # print(f"Found suitable flux offset: {phi:.4f} (avg g: {avg_g:.4f})")
                return phi

            phi += phi_step

        print(f"Warning: Could not find suitable flux offset. Max phi reached: {phi_max}")
        return phi_max


    def set_initial_flux_offset(self):
        """
        Finds a good initial flux offset and sets it for all nodes.
        """
        flux_offset = self.find_initial_flux_offset()
        self.flux_offset.data.fill_(flux_offset)
        










#

