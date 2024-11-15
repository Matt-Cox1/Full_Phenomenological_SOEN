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

# def update_state_minGRU(s,phi_old, phi,phi_z, g, gamma, tau, dt, clip_state,phi_final ):

#     phi_final = (1-phi_z)*phi_old + phi_z*phi
#     g_value = g(phi_final, s)
#     ds = gamma * g_value.detach() + (g_value - g_value.detach()) - s / tau
#     if clip_state:
#         s = torch.clamp(s + dt * ds, 0.0, 1)
#     else:
#         s = s + dt * ds
#     return s


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
    """
    This is a wrapper class for the above RateNN class. The idea behind it, is we want the neural network to act just
    like a frozen plug and play type function, just like the gaussian or sigmoid mixture functions. So we need to freeze all
    potentially learnable parameters, process the inputs into a form we trained the model on, and then pass it through the
    network.
    """
    def __init__(self, model_path, i_b):
        self.nn_model = RateNN()
        self.nn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        self.nn_model.eval()  
        self.i_b = i_b
        
        for param in self.nn_model.parameters():
            param.requires_grad = False
        
        self.periodic_function = make_periodic(self._original_call)

    def to(self, device):
        self.nn_model = self.nn_model.to(device)
        return self

    def _original_call(self, phi, s):
        device = phi.device
        i_b_tensor = torch.full_like(s, self.i_b, device=device) # if we ever want each node to have a different i_b the this needs to change
        input_data = torch.stack((s, phi.abs(), i_b_tensor), dim=-1)
        output = self.nn_model(input_data)
        result = output.squeeze(-1).clamp(min=0, max=1) # this step might not be needed but just in case
        return result

    def __call__(self, phi, s):
        return self.periodic_function(phi, s)
    
class NNDendriteWrapperMinGRU:
    """
    Similar to NNDendriteWrapper but for the MinGRU model
    """
    def __init__(self, model_path, i_b):
        self.nn_model = RateNN()
        self.nn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
        self.nn_model.eval()  
        self.i_b = i_b
        
        for param in self.nn_model.parameters():
            param.requires_grad = False
        
        self.periodic_function = make_periodic(self._original_call)

    def to(self, device):
        self.nn_model = self.nn_model.to(device)
        return self

    def _original_call(self, phi, s):
        device = phi.device
        i_b_tensor = torch.full_like(s, self.i_b, device=device) # if we ever want each node to have a different i_b the this needs to change
        input_data = torch.stack((s, phi.abs(), i_b_tensor), dim=-1)
        output = self.nn_model(input_data)
        result = output.squeeze(-1).clamp(min=0, max=1) # this step might not be needed but just in case
        return result
    




    def __call__(self, phi, s):
        return self.periodic_function(phi, s)

def make_periodic(g_function):
    """
    The source function reduces down to the region s=0 to s=1, and phi=0 to phi=0.5, and then becomes periodic in
    phi with a period of 1 and a reflection about phi=0.5.
    This function wraps around whatever source function (activation function) we chose to feed it and outputs another
    function that is periodic in phi.


    The make_periodic function is designed to take any given function g_function and transform 
    it into a new function that is periodic in the variable phi. 
    This means that the new function will repeat its values in a regular pattern over intervals of phi.
    """

    def periodic_g(phi, s, *args, **kwargs):
        
        # reflecting about phi=0
        phi_abs = torch.abs(phi)
        
        # restricting phi to live in \[0, 1) range (using the modulo operator)
        #  For example, phi=1.2 becomes 0.2
        phi_shifted = torch.fmod(phi_abs, 1.0)
        
        # Reflection: boolean mask for wherever phi_shifted is less than or equal to 0.5 
        mask = phi_shifted <= 0.5
        
        # apply the function or its reflection based on the mask
        phi_periodic = torch.where(mask, phi_shifted, 1 - phi_shifted)
        
        return g_function(phi_periodic, s, *args, **kwargs)
    
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


#From the auxiliary weight matrix JZ, find phi_z 
def calculate_phi_z(s, JZ_masked, flux_offset_Z, external_flux=None):
    phi = torch.mm(s, JZ_masked) + flux_offset_Z
    if external_flux is not None:
        phi += external_flux
    phi=torch.sigmoid(phi)
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










