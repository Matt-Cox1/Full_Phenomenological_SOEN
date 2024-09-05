
"""
This script uses the data from the rate arrays but only after they have been processed (converted from pickle file to csv)
and interpolated etc. Currently it is set up to take 3 inputs and output a single value. This can be easily changed 
in the TrainingConfig.
"""



import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
from dataclasses import dataclass

device = torch.device("cpu")


@dataclass # (dataclass is a decorator that automatically generates special methods - just makes it a bit neater)
class TrainingConfig:
    '''
    Any settings you wish to change about the learning process 
    '''
    num_epochs: int = 30
    start_lr: float = 0.01
    end_lr: float = 0.0001
    batch_size: int = 512

    input_size: int = 3
    hidden_size: int = 64
    output_size: int = 1

    train_split: float = 0.7
    val_split: float = 0.15

class RateDataset(Dataset):
    def __init__(self, features, targets): # both np arrays
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets.reshape(-1, 1)) # the -1 is to infer the number of rows

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



class RateNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RateNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


def load_df_from_pickle(file_path, compress=False):
    if compress:
        df = pd.read_pickle(file_path, compression='gzip')
    else:
        df = pd.read_pickle(file_path)
    
    print(f"DataFrame loaded from {file_path}")
    print(f"Shape: {df.shape}")
    return df


    
# def load_and_preprocess_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         if df.empty:
#             raise ValueError(f"The CSV file is empty: {file_path}")
        
#         required_columns = ['s', 'phi', 'ib', 'r_fq']
#         if not all(col in df.columns for col in required_columns):
#             raise ValueError(f"Missing required columns in the CSV file. Expected columns: {required_columns}")
        
#         features = df[['s', 'phi', 'ib']].values
#         targets = df['r_fq'].values
        
#         return features, targets
#     except FileNotFoundError:
#         raise FileNotFoundError(f"The CSV file was not found: {file_path}")
#     except Exception as e:
#         raise Exception(f"An error occurred while reading the CSV file: {str(e)}")

def load_and_preprocess_data(file_path, compress=True):
    try:
        df = load_df_from_pickle(file_path, compress)
        if df.empty:
            raise ValueError(f"The pickle file is empty: {file_path}")
        
        required_columns = ['s', 'phi', 'ib', 'r_fq']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns in the pickle file. Expected columns: {required_columns}")
        
        features = df[['s', 'phi', 'ib']].values
        targets = df['r_fq'].values
        
        return features, targets
    except FileNotFoundError:
        raise FileNotFoundError(f"The pickle file was not found: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the pickle file: {str(e)}")



def create_data_loaders(dataset, config):
    train_size = int(config.train_split * len(dataset))
    val_size = int(config.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    print(f"Dataset Summary:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader



def train_model(model, train_loader, val_loader, config, save_dir, csv_path):
    
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.start_lr)

    lr_lambda = lambda epoch: (config.end_lr / config.start_lr) ** (epoch / config.num_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')

    df = pd.DataFrame(columns=['Epoch', 'Train_Loss', 'Val_Loss', 'Learning_Rate'])
    df.to_csv(csv_path, index=False)

    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        

        for inputs, targets in train_loader:
            # Move inputs and targets to device (batch-wise)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # inputs on device
            loss = criterion(outputs, targets)  # targets on device
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch+1}/{config.num_epochs}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}, LR: {current_lr:.8f}')

        df = pd.DataFrame({'Epoch': [epoch+1], 'Train_Loss': [avg_train_loss], 'Val_Loss': [avg_val_loss], 'Learning_Rate': [current_lr]})
        df.to_csv(csv_path, mode='a', header=False, index=False)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

        scheduler.step()

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    df = pd.DataFrame({'Epoch': ['Training_Time'], 'Train_Loss': [training_time], 'Val_Loss': [None], 'Learning_Rate': [None]})
    df.to_csv(csv_path, mode='a', header=False, index=False)

    return train_losses, val_losses




def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    model.to(device)

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move inputs and targets to device
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.8f}')
    return test_loss



def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(12, 8), dpi=200)
    sns.set_style("white")
    sns.set_palette("deep")
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss', fontsize=22, fontweight='bold')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=17)
    
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {save_path}")

def main():
    config = TrainingConfig()
    
    try:
        file_path = 'experiments/source_functions/interpolated_rate_array_dendrite.pkl.gz'  # Path to the rate array pickle file
        features, targets = load_and_preprocess_data(file_path, compress=True)
        
        dataset = RateDataset(features, targets)
        train_loader, val_loader, test_loader = create_data_loaders(dataset, config)
        
        model = RateNN(input_size=config.input_size, hidden_size=config.hidden_size, output_size=config.output_size).to(device)
        
        save_dir = 'experiments/source_functions/trained_models/64_hidden_units'
        csv_path_logging = 'experiments/source_functions/trained_models/training_log.csv'
        
        train_losses, val_losses = train_model(model, train_loader, val_loader, config, save_dir, csv_path_logging)
        
        plot_losses(train_losses, val_losses, './loss_plot.png')
        
        test_loss = test_model(model, test_loader)

        df = pd.DataFrame({'Epoch': ['Test_Loss'], 'Train_Loss': [test_loss], 'Val_Loss': [None], 'Learning_Rate': [None]})
        df.to_csv(csv_path_logging, mode='a', header=False, index=False)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()





