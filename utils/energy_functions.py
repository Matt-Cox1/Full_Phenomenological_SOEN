# FILENAME: utils/energy_functions.py



import torch


def energy_kinetic(model, s, phi):
    """
    Calculate the kinetic energy of the system.
    """
    ds_dt = model.gamma * model.g(phi, s) - s / model.tau
    kinetic_energy = 0.5 * torch.sum(ds_dt**2, dim=1)
    return kinetic_energy

def energy_soen_global(model, s, phi, num_trapz_points=100):
    """
    E SOEN 1
    E_\text{global} = -\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n s_i J_{ij} s_j - \sum_{i=1}^n \gamma_i G_i(\phi_i, s_i) + \frac{1}{2} \sum_{i=1}^n \frac{s_i^2}{\tau_i}
    """

    batch_size, num_nodes = s.shape

    # Connection energy term
    connection_energy = -0.5 * torch.sum(s @ model.J * s, dim=1)

    # Activation integral term
    s_vals = torch.linspace(0, 1, num_trapz_points).unsqueeze(0).unsqueeze(0)
    s_vals = s_vals.expand(batch_size, num_nodes, -1)
    s_vals = s_vals * s.unsqueeze(-1)
    phi_expanded = phi.unsqueeze(-1).expand_as(s_vals)

    g_vals = model.g(phi_expanded, s_vals)
    G = torch.trapezoid(g_vals, s_vals, dim=-1)

    activation_energy = -torch.sum(model.gamma.unsqueeze(0) * G, dim=1)

    decay_energy = torch.sum(0.5 * s**2 / model.tau, dim=1)

    # Total global energy
    E_global = connection_energy + activation_energy + decay_energy 

    return E_global

def energy_soen_2(model, s, phi, num_trapz_points=100, epsilon=1e-8):
    """
    E SOEN 2
    """
    batch_size, num_nodes = s.shape

    # Calculate ds/dt using the SOEN dynamics equation
    ds_dt = model.gamma * model.g(phi, s) - s / model.tau

    # Create a tensor of integration points for each node
    s_vals = torch.linspace(0, 1, num_trapz_points).unsqueeze(0).unsqueeze(0)
    s_vals = s_vals.expand(batch_size, num_nodes, -1)
    s_vals = s_vals * s.unsqueeze(-1)

    # Expand phi to match s_vals dimensions
    phi_expanded = phi.unsqueeze(-1).expand_as(s_vals)

    # Calculate g values for all points
    g_vals = model.g(phi_expanded, s_vals)

    # Perform the trapezoidal integration to get G(\phi, s)
    G = torch.trapezoid(g_vals, s_vals, dim=-1)

    # Calculate the energy components with added epsilon for stability
    kinetic_term = 0.5 * (ds_dt**2)
    potential_term = -model.gamma * G
    dissipative_term = (s**2) / (2 * model.tau)

    # Compute the energy for each node
    E_node = torch.clamp(kinetic_term + potential_term + dissipative_term, min=epsilon)

    # Sum across all nodes to get the total energy
    E_total = E_node.sum(dim=1)

    return E_total

def energy_soen_3(model, s, phi, num_trapz_points=100, epsilon=1e-8):
    """
    3rd SOEN E
    """
    batch_size, num_nodes = s.shape

    # Calculate ds/dt using the SOEN dynamics equation
    ds_dt = model.gamma * model.g(phi, s) - s / model.tau

    # Create a tensor of integration points for each node
    s_vals = torch.linspace(0, 1, num_trapz_points).unsqueeze(0).unsqueeze(0)
    s_vals = s_vals.expand(batch_size, num_nodes, -1)
    s_vals = s_vals * s.unsqueeze(-1)

    # Expand phi to match s_vals dimensions
    phi_expanded = phi.unsqueeze(-1).expand_as(s_vals)

    # Calculate g values for all points
    g_vals = model.g(phi_expanded, s_vals)

    # Perform the trapezoidal integration to get G(\phi, s)
    G = torch.trapezoid(g_vals, s_vals, dim=-1)

    # Calculate the energy components with added epsilon for stability
    kinetic_term = 0.5 * (ds_dt**2)
    potential_term = -model.gamma * G
    dissipative_term = (s**2) / (2 * model.tau)

    # Compute the energy for each node
    E_node = torch.clamp(kinetic_term + potential_term + dissipative_term, min=epsilon)

    # Sum across all nodes to get the total energy
    E_total = E_node.sum(dim=1)

    connection_energy = (0.5 * torch.sum(s @ model.J * s, dim=1))

    return E_total + connection_energy

def energy_soen_4(model, s, phi, num_trapz_points=100):
    """
    E SOEN 4
    """
    batch_size, num_nodes = s.shape

    ds_dt = model.gamma * model.g(phi, s) - s / model.tau
    kinetic_term = torch.sum(0.5*ds_dt**2,dim=1)

    # Connection energy term
    connection_energy = -0.5 * torch.sum(s @ model.J * s, dim=1)

    # Activation integral term
    s_vals = torch.linspace(0, 1, num_trapz_points).unsqueeze(0).unsqueeze(0)
    s_vals = s_vals.expand(batch_size, num_nodes, -1)
    s_vals = s_vals * s.unsqueeze(-1)
    phi_expanded = phi.unsqueeze(-1).expand_as(s_vals)

    g_vals = model.g(phi_expanded, s_vals)
    G = torch.trapezoid(g_vals, s_vals, dim=-1)

    activation_energy = -torch.sum(model.gamma.unsqueeze(0) * G, dim=1)

    # State decay energy term
    decay_energy = torch.sum(0.5 * s**2 / model.tau, dim=1)

    # Total global energy
    E_global = connection_energy + activation_energy + decay_energy + kinetic_term

    return E_global

def energy_soen_local(model, s, phi, num_trapz_points=100):
    """
    E SOEN LOCAL
    """
    batch_size, num_nodes = s.shape

    # we will vectorise operations to speed this up a little
    # create a tensor of integration points for each node
    s_vals = torch.linspace(0, 1, num_trapz_points).unsqueeze(0).unsqueeze(0)
    s_vals = s_vals.expand(batch_size, num_nodes, -1)
    s_vals = s_vals * s.unsqueeze(-1)
    # expand phi to match s_vals dimensions
    phi_expanded = phi.unsqueeze(-1).expand_as(s_vals)

    # calculate g values for all points
    g_vals = model.g(phi_expanded, s_vals)

    # perform the trapezoidal integration
    G = torch.trapezoid(g_vals, s_vals, dim=-1)

    # energy value for a single node (all nodes done in parallel because this is vectorised)
    E = -model.gamma * G + (0.5 / model.tau) * s**2

    # Finally, sum across all nodes
    E = E.sum(dim=1)

    return E

def energy_sam(model, s, phi):
    """
    Calculate the energy using Sam's formula:
    E_{Sam} = 1/2 \tau\gamma g^T J g - \phi^{ext} g
    """
    g = model.g(phi, s)  # Shape: (batch_size, num_total)
    J_masked = model.J * model.mask  # Shape: (num_total, num_total)
    
    # Calculate g^T J g for each item in the batch
    gJg = torch.bmm(g.unsqueeze(1), torch.matmul(J_masked, g.unsqueeze(2))).squeeze()

    E_sam = 0.5 * torch.sum(model.tau * model.gamma, dim=0) * gJg - torch.sum(model.external_flux * g, dim=1)

    E_sam = E_sam / (model.num_total)

    return E_sam

def energy_hopfield(model, s, phi):
    """
    Calculate the energy using the Hopfield formula from the EP paper but with phi as the membrane potential as described
    in the 2024 p. model paper
    """
    g = model.g(phi, s)
    J_masked = model.J * model.mask
    

    E1 = 0.5 * torch.sum(phi**2, dim=1)
    
    # Second term: 
    E2 = -0.5 * (torch.einsum('bi,ij,bj->b', g, J_masked, g))
    
    # third term:
    E3 = -torch.sum((model.flux_offset+model.external_flux) * g, dim=1)
    
    E_hopfield = E1 + E2 + E3
    
    return E_hopfield

def energy_simple(model, s, phi):
    """
    A really simple sum of the squares of the states. 
    """
    return torch.sum(s**2, dim=1)