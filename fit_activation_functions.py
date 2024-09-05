import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
import numpy as np
import ast

from model.soen_model import SOENModel
from model.model_config import SOENConfig


"""
The goal of this python script is to assess the suitability of proxy activation/source functions in terms of their ability
to be fit to the rate array. (The rate array we're assuming for the purpose of this script is close enough, so as to be equivilent,to
the neural network we trained on it)

- This script first tries to fit a Gaussian Mixture and Sigmoid Mixture function to the true source function using scipy's curve_fit
for every different bias current stored in the rate array (15 unique dendritic bias currents were originally simulated) - this of course 
can be flexible later down the line when we instead model a neuron source function which would take in 2 different bias currents.
- Then, given that we now have the best parameters (that curve_fit could find) for each function and for each bias current, 
the idea was to use the best params (over all bias currents) for each function to fit the rate array again during a second sweep.
My motivation for doing this, was that the initial conditions turned out to be very important for the curve_fit to converge to a good solution.
- The multiple sweep technique worked very well, it increased both the stability of the convergence and the quality of the fit.
- It seems from this short experiment that the gaussian mixture function was the best fit (at least for this particular rate array)


"""

# ======================== Data Generation ========================

def generate_nn_data(i_b, resolution=200):
    """
    Generate data from the neural network model of the rate array we chose to model
    """
    config = SOENConfig(activation_function="NN_dendrite")
    model = SOENModel(config)
    model.i_b = i_b
    model._initialise_nn_dendrite()

    s_values = torch.linspace(0, 1, resolution)
    phi_values = torch.linspace(0, 0.5, resolution)
    s_grid, phi_grid = torch.meshgrid(s_values, phi_values, indexing='ij')

    with torch.no_grad():
        g_output = model.g(phi_grid.flatten(), s_grid.flatten()).reshape(resolution, resolution)

    return s_grid, phi_grid, g_output

# ======================== Fitting Functions ========================

def gaussian_mixture(params, s, phi):
    """
    Gaussian mixture function for fitting.
    """
    A, mu_s, sigma_s, mu_phi, sigma_phi, B, C, D, E, F, G = params
    
    gaussian_s = torch.exp(-((s - mu_s)**2 / (2 * sigma_s**2)))
    gaussian_phi = torch.exp(-((phi - mu_phi)**2 / (2 * sigma_phi**2)))
    s_decay = torch.exp(-B * (s - 1)**2)
    phi_modulation = 1 / (1 + torch.exp(-C * (phi - D)))
    
    return torch.clamp(A * gaussian_s * gaussian_phi * s_decay * phi_modulation + E * s + F * phi + G, 0, 1)

def sigmoid_mixture(params, s, phi):
    """
    Sigmoid mixture function for fitting.
    """
    A1, mu_s, sigma_s, A2, mu_phi, sigma_phi, B, C, D, E = params
    
    sigmoid_s = A1 / (1 + torch.exp(-(s - mu_s) / sigma_s))
    sigmoid_phi = A2 / (1 + torch.exp(-(phi - mu_phi) / sigma_phi))
    scaling = 1 / (1 + torch.exp(-B * (s + C * phi - D)))
    
    return torch.clamp((sigmoid_s + sigmoid_phi) * scaling + E, 0, 1)



# ======================== Optimisation Functions ========================

def error_function(params, func, s, phi, g_output):
    """
    Calculate the mean squared error between the fitted function and the actual output.
    """
    params_tensor = torch.tensor(params, dtype=torch.float32)
    fitted = func(params_tensor, s, phi)
    return torch.mean((g_output - fitted)**2).item()

def fit_function(func, s_grid, phi_grid, g_output, initial_params):
    """
    Fit the given function to the data using optimisation.
    """
    result = minimize(
        error_function,
        initial_params,
        args=(func, s_grid.flatten(), phi_grid.flatten(), g_output.flatten()),
        method='Nelder-Mead',
        options={'maxiter': 100_000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    return result.x

# ======================== Visualization Functions ========================

def plot_results(s_grid, phi_grid, g_output, fitted_surface, i_b, func_name, output_dir):
    """
    Plot the original data, fitted surface, and their difference.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    for ax, data, title in zip([ax1, ax2, ax3],
                               [g_output, fitted_surface, torch.abs(g_output - fitted_surface)],
                               ['Original NN g(s, phi)', f'Fitted g(s, phi) - {func_name}', 'Absolute Difference']):
        im = ax.contourf(s_grid.numpy(), phi_grid.numpy(), data.numpy(), cmap='viridis', levels=50)
        ax.set_title(title)
        ax.set_xlabel('s')
        ax.set_ylabel('phi')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fit_results_i_b_{i_b}_{func_name}.png'))
    plt.close()

def create_summary_plots(results_df, output_dir, suffix=''):
    """
    Create summary plots for all results.
    """
    # MSE vs i_b plot
    plt.figure(figsize=(12, 8))
    for name in results_df['function_name'].unique():
        data = results_df[results_df['function_name'] == name]
        plt.plot(data['i_b'], data['mse'], marker='o', label=name)
    plt.xlabel('Dendrite Bias Current (i_b)')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.legend(title='Function')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mse_vs_i_b{suffix}.png'))
    plt.close()


# ======================== Main Processing Functions ========================

def process_single_ib(i_b, initial_params, output_dir, sweep_num):
    """
    Process a single i_b value: generate data, fit functions, and plot results.
    """
    print(f"\nProcessing i_b = {i_b} (Sweep {sweep_num})")
    s_grid, phi_grid, g_output = generate_nn_data(i_b)
    
    results = []
    functions = {
        'Gaussian Mixture': (gaussian_mixture, initial_params['gaussian']),
        'Sigmoid Mixture': (sigmoid_mixture, initial_params['sigmoid'])
    }
    
    for func_name, (func, init_params) in functions.items():
        best_params = fit_function(func, s_grid, phi_grid, g_output, init_params)
        fitted_surface = func(torch.tensor(best_params), s_grid, phi_grid)
        mse = torch.mean((g_output - fitted_surface)**2).item()
        
        print(f"MSE for {func_name}: {mse:.6e}")
        plot_results(s_grid, phi_grid, g_output, fitted_surface, i_b, f"{func_name}_sweep{sweep_num}", output_dir)
        results.append((func_name, mse, str(best_params.tolist())))
    
    return results

def perform_sweep(i_b_values, initial_params, output_dir, sweep_num):
    """
    Perform a single sweep across all i_b values.
    """
    all_results = []
    for i_b in i_b_values:
        results = process_single_ib(i_b, initial_params, output_dir, sweep_num)
        all_results.extend([(i_b, *r) for r in results])
    
    return pd.DataFrame(all_results, columns=['i_b', 'function_name', 'mse', 'parameters'])

def run(i_b_values, initial_params, output_dir):
    """
    Main function to run the entire process for all i_b values with two sweeps.
    """
    os.makedirs(output_dir, exist_ok=True)

    # First sweep
    print("Performing first sweep...")
    results_df_1 = perform_sweep(i_b_values, initial_params, output_dir, 1)
    results_df_1.to_csv(os.path.join(output_dir, 'fitting_results_sweep1.csv'), index=False)
    create_summary_plots(results_df_1, output_dir, suffix='_sweep1')

    # Find best i_b for each function based on MSE
    best_params = {}
    for func_name in results_df_1['function_name'].unique():
        func_results = results_df_1[results_df_1['function_name'] == func_name]
        best_i_b = func_results.loc[func_results['mse'].idxmin(), 'i_b']
        best_params[func_name] = eval(func_results[func_results['i_b'] == best_i_b]['parameters'].values[0])

    # Prepare initial params for second sweep
    second_initial_params = {
        'gaussian': best_params['Gaussian Mixture'],
        'sigmoid': best_params['Sigmoid Mixture'],
    }

    # Second sweep
    print("\nPerforming second sweep with optimized initial parameters...")
    results_df_2 = perform_sweep(i_b_values, second_initial_params, output_dir, 2)
    results_df_2.to_csv(os.path.join(output_dir, 'fitting_results_sweep2.csv'), index=False)
    create_summary_plots(results_df_2, output_dir, suffix='_sweep2')

    # Compare results from both sweeps
    compare_sweeps(results_df_1, results_df_2, output_dir)

    print(f"\nAll operations completed. Results saved in: {output_dir}")

def compare_sweeps(results_df_1, results_df_2, output_dir):
    """
    Compare and visualise the results from both sweeps with improved clarity and specific styling.
    """
    plt.figure(figsize=(12, 8))
    
    # colours for each function
    colors = {'Gaussian Mixture': 'black', 'Sigmoid Mixture': 'red'}
    
    for name in results_df_1['function_name'].unique():
        data_1 = results_df_1[results_df_1['function_name'] == name]
        data_2 = results_df_2[results_df_2['function_name'] == name]
        
        plt.plot(data_1['i_b'], data_1['mse'], color=colors[name], linestyle='-', 
                 marker='o', markersize=6, label=f'{name} (Sweep 1)')
        plt.plot(data_2['i_b'], data_2['mse'], color=colors[name], linestyle='--', 
                 marker='s', markersize=6, label=f'{name} (Sweep 2)')

    plt.xlabel('Dendritic Bias Current ($i_b$)', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.yscale('log')
    

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=14, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_comparison_between_sweeps.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # improvement percentages
    improvement_data = []
    for name in results_df_1['function_name'].unique():
        for i_b in results_df_1['i_b'].unique():
            mse_1 = results_df_1[(results_df_1['function_name'] == name) & (results_df_1['i_b'] == i_b)]['mse'].values[0]
            mse_2 = results_df_2[(results_df_2['function_name'] == name) & (results_df_2['i_b'] == i_b)]['mse'].values[0]
            improvement = (mse_1 - mse_2) / mse_1 * 100
            improvement_data.append([name, i_b, improvement])

    improvement_df = pd.DataFrame(improvement_data, columns=['function_name', 'i_b', 'improvement_percentage'])
    
    plt.figure(figsize=(12, 8))
    for name in improvement_df['function_name'].unique():
        data = improvement_df[improvement_df['function_name'] == name]
        plt.plot(data['i_b'], data['improvement_percentage'], color=colors[name], 
                 marker='o', markersize=6, label=name)
    
    plt.xlabel('Dendrite Bias Current (i_b)', fontsize=12)
    plt.ylabel('Improvement Percentage', fontsize=12)
    plt.title('Improvement in MSE from Sweep 1 to Sweep 2', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(title='Function', title_fontsize=12, 
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_improvement_percentage.png'), dpi=300, bbox_inches='tight')
    plt.close()



# ======================== Script Execution ========================

if __name__ == "__main__":

    # Define i_b values to process
    i_b_values = [1.3524, 1.4024, 1.4524, 1.5024, 1.5524, 1.6024, 1.6524, 1.7024, 1.7524, 1.8024, 1.8524, 1.9024, 1.9524, 2.0024, 2.0524]
    
    # Initial parameters for each fitting function
    initial_params = {
        'gaussian': [0.5, 0.5, 0.2, 0.25, 0.1, 10.0, 20.0, 0.25, 0.0, 0.0, 0.0],
        'sigmoid': [-50.0, -3000.0, -5000.0, -30.0, -0.1, -0.05, 0.1, -0.5, -1.0, 10.0],
    }
    
    # Output directory for results
    output_dir = 'Results/fit_activation_functions'
    
    # Run the main process
    run(i_b_values, initial_params, output_dir)








############################






def compare_sweeps_from_csv(csv_file1, csv_file2, output_dir, use_log_scale=False):
    """
    Load CSV files, create a comparison plot of the results from both sweeps,
    and generate a new CSV with only the lowest MSE values and their corresponding parameters.
    
    Parameters:
    - csv_file1: Path to the CSV file for sweep 1
    - csv_file2: Path to the CSV file for sweep 2
    - output_dir: Directory to save the output plot and new CSV
    - use_log_scale: Boolean to toggle between log and linear scale for y-axis
    """
    # Load CSV files
    results_df_1 = pd.read_csv(csv_file1)
    results_df_2 = pd.read_csv(csv_file2)
    

    plt.figure(figsize=(12, 8),dpi=200)
    colors = {'Gaussian Mixture': 'black', 'Sigmoid Mixture': 'red'}
    
    # Create a new dataframe to store only the lowest MSE values and parameters
    lowest_mse_df = pd.DataFrame(columns=['i_b', 'function_name', 'mse', 'sweep', 'parameters'])
    
    for name in results_df_1['function_name'].unique():
        data_1 = results_df_1[results_df_1['function_name'] == name]
        data_2 = results_df_2[results_df_2['function_name'] == name]
        
        plt.plot(data_1['i_b'], data_1['mse'], color=colors[name], linestyle='-', 
                 marker='o', markersize=6, label=f'{name} (Sweep 1)')
        plt.plot(data_2['i_b'], data_2['mse'], color=colors[name], linestyle='--', 
                 marker='s', markersize=6, label=f'{name} (Sweep 2)')
        
        # Compare MSE values and keep only the lowest for each i_b
        for i_b in data_1['i_b']:
            mse_1 = data_1[data_1['i_b'] == i_b]['mse'].values[0]
            mse_2 = data_2[data_2['i_b'] == i_b]['mse'].values[0]
            
            if mse_1 <= mse_2:
                lowest_mse = data_1[data_1['i_b'] == i_b].copy()
                lowest_mse['sweep'] = 1
            else:
                lowest_mse = data_2[data_2['i_b'] == i_b].copy()
                lowest_mse['sweep'] = 2
            
            lowest_mse_df = pd.concat([lowest_mse_df, lowest_mse[['i_b', 'function_name', 'mse', 'sweep', 'parameters']]])

    plt.xlabel('Dendritic Bias Current ($i_b$)', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    
    if use_log_scale:
        plt.yscale('log')
    

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=14, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_comparison_between_sweeps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convert parameters string to list
    lowest_mse_df['parameters'] = lowest_mse_df['parameters'].apply(ast.literal_eval)
    
    # Save only the lowest MSE values and their corresponding parameters to a new CSV file
    lowest_mse_df.to_csv(os.path.join(output_dir, 'lowest_mse_values_and_params.csv'), index=False)
    print(f"Lowest MSE values and parameters saved to: {os.path.join(output_dir, 'lowest_mse_values_and_params.csv')}")


if __name__ == "__main__":
    csv_file1 = 'Results/fit_activation_functions/fitting_results_sweep1.csv'
    csv_file2 = 'Results/fit_activation_functions/fitting_results_sweep2.csv'
    output_dir = 'Results/fit_activation_functions/'
    
    compare_sweeps_from_csv(csv_file1, csv_file2, output_dir, use_log_scale=False)

















# # #