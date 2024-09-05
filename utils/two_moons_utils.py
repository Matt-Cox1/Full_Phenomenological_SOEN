# File: Application/utils/two_moons_utils.py

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import torch

def prepare_two_moons_data(n_samples=1000, noise=0.1, test_size=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X))
        _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == torch.LongTensor(y)).float().mean().item()
    return accuracy

def get_decision_boundary(model, X, y, res=0.1):
    model.eval()
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_tensor = torch.FloatTensor(grid_points)
    
    with torch.no_grad():
        outputs = model(grid_points_tensor)
        predicted = outputs.argmax(dim=1).numpy()
    
    predicted = predicted.reshape(xx.shape)
    
    return {
        'xx': xx,
        'yy': yy,
        'predicted': predicted,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'X': X,
        'y': y
    }