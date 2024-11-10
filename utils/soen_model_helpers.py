# FILENAME: utils/soen_model_helpers.py

import torch
import torch.nn as nn
import numpy as np
import random


def update_state(s, phi, g, gamma, tau, dt, clip_state):
    g_value = g(phi, s)
    ds = gamma * g_value.detach() + (g_value - g_value.detach()) - s / tau
    if clip_state:
        s = torch.clamp(s + dt * ds, 0.0, 1)
    else:
        s = s + dt * ds
    return s


class RateNN(nn.Module):
    """
    This is a small neural network (should be the exact same as the one used to train the NN_dendrite - n.t.s I should probably refactor the code
    to only use one and import it)
    - But this neural network is used to approximate (store information about) the rate of fluxon prodcuction of
    whichever dendrite or soma we choose to model
    """
    def __init__(self):
        super(RateNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        So whenever we lets say write model(x), the forward function is called and x is passed through the network
        """
        return self.network(x)


class NNDendriteWrapper:
    def __init__(self, model_path, i_b):
        self.nn_model = RateNN()
        self.nn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.nn_model.eval()  
        self.i_b = i_b
            
        for param in self.nn_model.parameters():
            param.requires_grad = False
            
        self.periodic_function = make_periodic(self._original_call)
    
    def to(self, device):
        self.nn_model = self.nn_model.to(device)
        self.i_b = self.i_b.to(device)
        return self
    
    def _original_call(self, phi, s, i_b_relevant):
        # Ensure all inputs are on the same device
        device = phi.device
        s = s.to(device)
        i_b_relevant = i_b_relevant.to(device)

        # Ensure all inputs are the same shape
        # Inputs are (batch_size, num_relevant_nodes)
        # Stack along the last dimension to create input of shape (batch_size, num_relevant_nodes, 3)
        input_data = torch.stack((s, phi.abs(), i_b_relevant), dim=-1)
        # Reshape to (batch_size * num_relevant_nodes, 3) for the model
        input_data = input_data.view(-1, 3)
        output = self.nn_model(input_data)
        # Reshape back to (batch_size, num_relevant_nodes)
        result = output.view(s.shape).clamp(min=0, max=1)
        return result
    
    def __call__(self, phi, s, i_b_relevant):
        return self.periodic_function(phi, s, i_b_relevant)



def make_periodic(g_function):
    def periodic_g(phi, s, *args, **kwargs):
        # Ensure phi is within [-0.5, 0.5]
        phi = torch.remainder(phi + 0.5, 1.0) - 0.5
        return g_function(phi.abs(), s, *args, **kwargs)
    return periodic_g



def initialise_state(batch_size, num_total, initial_state, device, clip_state):
    if initial_state is None:
        s = torch.zeros(batch_size, num_total, device=device)
    else:
        s = initial_state.to(device)
        if clip_state:
            s = torch.clamp(s, 0.0, 1)
    return s

def apply_input(s, x, input_nodes, input_type, is_input_time_varying, time_step=None):
    '''
    This function either sets the external flux values or clamps the state values of the input nodes.
    If we're feeding input to the output nodes then `input_nodes` would be the output nodes. (Sorry, that just sounds 
    confusing)
    '''
    if is_input_time_varying:
        if input_type == 'state':
            s[:, input_nodes] = x[:, :, time_step]
        elif input_type == 'flux':
            external_flux = torch.zeros_like(s)
            external_flux[:, input_nodes] = x[:, :, time_step]
            return external_flux
    else:
        if input_type == 'state':
            s[:, input_nodes] = x
        elif input_type == 'flux':
            external_flux = torch.zeros_like(s)
            external_flux[:, input_nodes] = x
            return external_flux
    return None

def calculate_phi(s, J_masked, flux_offset, external_flux=None):
    phi = torch.mm(s, J_masked) + flux_offset 
    if external_flux is not None:
        phi += external_flux
    return phi

def apply_noise(phi, noise_std, training):
    """
    We could chose to apply noise from a different distribution, 
    but for now we are using a Gaussian distribution
    """
    if training:
        noise = torch.randn_like(phi) * noise_std
    else:
        # but if model.eval() then:
        noise = torch.randn_like(phi) * noise_std
    return phi + noise

def check_convergence(s, s_old, tol):
    return torch.max(torch.abs(s - s_old)) < tol










