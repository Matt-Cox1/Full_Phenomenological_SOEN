# FILENAME: utils/data_loader.py

from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from utils.two_moons_utils import prepare_two_moons_data

import audio_data.spoken_digits as spoken_digits





torch.manual_seed(42)
np.random.seed(42)

def load_mnist_data(batch_size=64, val_split=0.2):
    """
    Load and prepare MNIST dataset.
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split training data into train and validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    return train_loader, val_loader, test_loader





def load_two_moons_data(X=None, y=None, scaler=None, batch_size=32, val_split=0.2):
    if X is None or y is None or scaler is None:
        X, y, scaler = prepare_two_moons_data()

    # Split the data into train and validation sets
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    # Create TensorDatasets: these are needed for the DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create DataLoaders: The dataloaders handle the minibatching and shuffling of the data automatically.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader





def load_audio_data(batch_size=256, val_split=0.2):
    """
    Load and prepare spoken digits dataset.
    
    Args:
        batch_size (int): Size of each batch
        val_split (float): Fraction of data to use for validation
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) where each loader provides
        batches of spectrograms with shape (batch_size, 16, 1024) and labels
    """
    try:
        # Try to load pre-processed datasets first
        train_dataset, val_dataset, test_dataset = spoken_digits.load_processed_datasets('processed_data')
    except FileNotFoundError:
        # If not found, load and process the raw data
        examples = spoken_digits.load_audio_files(spoken_digits.base_dir)
        train_dataset, val_dataset, test_dataset = spoken_digits.prepare_data_splits(
            examples, 
            train_ratio=0.7,
            val_ratio=val_split
        )
        # Save the processed datasets for future use
        spoken_digits.save_processed_datasets(
            train_dataset, 
            val_dataset, 
            test_dataset, 
            'processed_data'
        )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get a sample batch to print format
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\nAudio Data Format:")
    print(f"Batch shape: {sample_batch.shape}")
    print(f"Labels shape: {sample_labels.shape}")
    print(f"Sample data type: {sample_batch.dtype}")
    print(f"Sample labels type: {sample_labels.dtype}\n")

    return train_loader, val_loader, test_loader