
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model.soen_model import SOENModel
from model.model_config import SOENConfig


"""
This could be another good method to initialise the flux offsets rather than finding the threshold directly,
we could place each dendrite at the maximum gradient of the activation function. This would mean each source function's
response would be maximally sensitive to changes in the input.

This results in a very similar flux value to that of the threshold method.

The benefit of this approach is that we get an optimal phi for every s. In this script we just average over s to get a single phi
value. However, this could be tied into a better flux offset initialisation method by using the incoming connections and 
the weight matrix to estimate the s value at each node. And then the flux offset would be tailor to the s value at each node.
"""

def compute_max_gradients(model, num_s_points=1000, num_phi_points=10000):
    # Create evenly spaced state values from 0 to 1
    s_values = torch.linspace(0, 1, num_s_points).unsqueeze(1)
    phi_values = torch.linspace(0, 0.5, num_phi_points).unsqueeze(0).repeat(num_s_points, 1)
    
    # Enable gradient computation
    phi_values.requires_grad_(True)
    
    # Compute g for each combination of s and phi
    g_values = model.g(phi_values, s_values.expand_as(phi_values))
    
    # Compute gradients using pyTorch's autograd
    gradients = torch.autograd.grad(g_values.sum(), phi_values)[0]
    
    # Find maximum gradient and corresponding phi for each s value
    max_gradients, max_indices = gradients.abs().max(dim=1)
    phi_at_max_grad = phi_values[0, max_indices]
    
    return max_gradients, phi_at_max_grad, s_values.squeeze()








config = SOENConfig()
config.activation_function = "NN_dendrite"
model = SOENModel(config)

# Compute maximum gradients and corresponding phi values
max_gradients, phi_at_max_grad, s_values = compute_max_gradients(model)

# Calculate and print the average phi value at maximum gradient
avg_phi_at_max_grad = phi_at_max_grad.mean().item()
print(f"Average phi value at maximum gradient: {avg_phi_at_max_grad:.4f}")

# Print phi value of max gradient at s=0 and s=1
print(f"Phi value of max gradient at s=0: {phi_at_max_grad[0].item():.4f}")
print(f"Phi value of max gradient at s=1: {phi_at_max_grad[-1].item():.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(s_values.detach(), phi_at_max_grad.detach(), label="Phi at max gradient")
plt.title("Phi value of maximum gradient vs s")
plt.xlabel("s")
plt.ylabel("Phi at max gradient")
plt.grid(True)
plt.show()

