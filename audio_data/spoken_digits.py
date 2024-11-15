# Import required libraries
import torch
import torchaudio
import os
import numpy as np
import sounddevice as sd
import time
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import pickle

# Define the base directory for the dataset
base_dir = Path("C:/Users/pseud/data/speech_commands/SpeechCommands/speech_commands_v0.02")

# Configuration constants
NUM_DIGITS = 10  # Total number of digits (0-9)
EXAMPLES_PER_DIGIT = 2048  # Default number of examples per digit

# Map digit labels to their folder names
digit_label_map = {
    i: name for i, name in enumerate(
        ["zero", "one", "two", "three", "four", 
         "five", "six", "seven", "eight", "nine"][:NUM_DIGITS]
    )
}

def load_audio_files(base_dir, examples_per_digit=EXAMPLES_PER_DIGIT):
    examples = []
    target_length = 16000  # 1 second at 16kHz sampling rate
    length_tolerance = int(0.3 * 16000)  # 300ms tolerance in samples
    
    # New RMS energy parameters
    tail_duration = int(0.2 * 16000)  # Last 200ms
    energy_threshold = 0.7  # 70% threshold
    
    for digit, folder_name in digit_label_map.items():
        digit_path = base_dir / folder_name
        all_audio_files = list(digit_path.glob("*.wav"))
        valid_files = []
        
        # First pass: check lengths and energy distribution
        for audio_file in all_audio_files:
            waveform, sample_rate = torchaudio.load(audio_file)
            waveform = waveform.squeeze()
            length_diff = abs(waveform.shape[0] - target_length)
            
            if length_diff <= length_tolerance:
                # Calculate RMS energy for the whole signal and the tail
                total_energy = torch.sqrt(torch.mean(waveform ** 2))
                tail_energy = torch.sqrt(torch.mean(waveform[-tail_duration:] ** 2))
                
                # Check if the tail energy isn't too dominant
                if tail_energy / (total_energy + 1e-6) <= energy_threshold:
                    valid_files.append(audio_file)
        
        # Rest of the function remains the same...
        selected_files = np.random.choice(
            valid_files, 
            size=min(examples_per_digit, len(valid_files)), 
            replace=False
        )
        
        print(f"Loading {len(selected_files)} examples for digit {folder_name} "
              f"(rejected {len(all_audio_files) - len(valid_files)} files for length/energy)")
        
        for audio_file in selected_files:
            waveform, sample_rate = torchaudio.load(audio_file)
            examples.append({
                "audio": {
                    "array": waveform.squeeze().numpy(),
                    "sampling_rate": sample_rate
                },
                "label": digit,
                "path": str(audio_file)
            })
    
    return examples

def create_mel_spectrogram(waveform, sample_rate, moving_average_params=None, derivative_params=None, use_db_scale=False):
    """Create mel spectrogram with specified parameters and optional post-processing.
    
    Args:
        waveform: Input audio waveform
        sample_rate: Audio sampling rate
        moving_average_params (dict, optional): Parameters for moving average filtering
        derivative_params (dict, optional): Parameters for normalized derivative
        use_db_scale (bool): Whether to convert to dB scale before processing
    """
    n_fft = int(sample_rate * 0.05)  # 25ms windows
    hop_length = int(sample_rate * 0.025)  # 20ms hop length
    
    # Calculate target length for 1 secon212d of audio
    target_length = int(sample_rate / hop_length)  # Number of frames for 1 second
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=32,
        f_min=50,
        f_max=7000,
        mel_scale='htk'
    )
    
    # Convert to tensor if it's not already
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    # Ensure input is 2D
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Normalize waveform to have a consistent peak value
    peak_value = 1.0
    waveform = waveform / waveform.abs().max() * peak_value
    
    # Create spectrogram
    mel_spec = mel_transform(waveform)
    
    # Convert to dB scale if requested
    if use_db_scale:
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        mel_spec = amplitude_to_db(mel_spec)
    
    # Normalize the values to 0-1 range
    mel_spec = mel_spec - mel_spec.min()
    mel_spec = mel_spec / (mel_spec.max() + 1e-6)
    
    # Pad or trim to target length
    current_length = mel_spec.shape[2]
    
    if current_length < target_length:
        # Pad
        padding = target_length - current_length
        mel_spec = F.pad(mel_spec, (0, padding))
    elif current_length > target_length:
        # Trim
        mel_spec = mel_spec[:, :, :target_length]
    
    # Remove the unnecessary first dimension
    mel_spec = mel_spec.squeeze(0)
    # print(f"Max value: {mel_spec.max().item()}")
    
    # Apply moving average if parameters are provided
    if moving_average_params is not None:
        mel_spec = apply_moving_average(
            mel_spec,
            window_size=moving_average_params.get('window_size', 3),
            alpha=moving_average_params.get('alpha', 0.3),
            method=moving_average_params.get('method', 'simple')
        )

    min_val = 0
    mel_spec = mel_spec + min_val
    
    # Apply normalized derivative if parameters are provided
    if derivative_params is not None:
        mel_spec = calculate_normalized_derivative(
            mel_spec,
            min_clip=derivative_params.get('min_clip', -5),
            max_clip=derivative_params.get('max_clip', 5),
            epsilon=derivative_params.get('epsilon', 1e-6)
        )
    
    return mel_spec

def apply_moving_average(time_series, window_size=3, alpha=0.3, method='simple'):
    """
    Apply moving average filter to a time series (spectrogram).
    
    Args:
        time_series (torch.Tensor): Input time series data (mel spectrogram)
        window_size (int): Size of the moving average window for SMA
        alpha (float): Smoothing factor for EMA (0 < alpha < 1)
        method (str): 'simple' for SMA or 'exponential' for EMA
    
    Returns:
        torch.Tensor: Smoothed time series data
    """
    if not isinstance(time_series, torch.Tensor):
        time_series = torch.tensor(time_series, dtype=torch.float32)
    
    if method.lower() == 'simple':
        # Create the averaging kernel
        kernel = torch.ones(1, 1, window_size) / window_size
        
        # Pad the input to maintain size
        padding = (window_size - 1) // 2
        
        # Process each frequency bin separately
        smoothed = torch.zeros_like(time_series)
        for i in range(time_series.shape[0]):
            # Apply convolution to each frequency bin
            smoothed[i:i+1] = F.conv1d(
                time_series[i:i+1].unsqueeze(0), 
                kernel,
                padding=padding
            ).squeeze(0)
        
    elif method.lower() == 'exponential':
        smoothed = time_series.clone()
        for t in range(1, time_series.shape[1]):
            smoothed[:, t] = alpha * time_series[:, t] + (1 - alpha) * smoothed[:, t-1]
    
    else:
        raise ValueError("Method must be either 'simple' or 'exponential'")
    
    return smoothed

class SpokenDigitsDataset(Dataset):
    def __init__(self, examples, preprocessed=False, moving_average_params=None, 
                 derivative_params=None, use_db_scale=False):
        """
        Args:
            examples: List of audio examples
            preprocessed: Whether the examples are already preprocessed
            moving_average_params (dict, optional): Parameters for moving average filtering
            derivative_params (dict, optional): Parameters for normalized derivative
            use_db_scale (bool): Whether to convert to dB scale before processing
        """
        self.examples = examples
        self.preprocessed = preprocessed
        self.use_db_scale = use_db_scale
        
        # Set default processing parameters
        self.moving_average_params = {
            'window_size': 3,
            'alpha': 0.3,
            'method': 'exponential'
        }
        if moving_average_params is not None:
            self.moving_average_params.update(moving_average_params)
        else:
            self.moving_average_params = None
            
        self.derivative_params = {
            'min_clip': -5,
            'max_clip': 5,
            'epsilon': 1e-6
        }
        if derivative_params is not None:
            self.derivative_params.update(derivative_params)
        else:
            self.derivative_params = None
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if self.preprocessed:
            return example['spectrogram'].squeeze(), example['label']
        else:
            mel_spec = create_mel_spectrogram(
                example['audio']['array'],
                example['audio']['sampling_rate'],
                moving_average_params=self.moving_average_params,
                derivative_params=self.derivative_params,
                use_db_scale=self.use_db_scale
            )
            
            return mel_spec, torch.tensor(example['label'], dtype=torch.long)

def prepare_data_splits(examples, train_ratio=0.7, val_ratio=0.15, 
                       moving_average_params=None, derivative_params=None, 
                       use_db_scale=False):
    """Split data into train, validation, and test sets."""
    dataset = SpokenDigitsDataset(
        examples,
        moving_average_params=moving_average_params,
        derivative_params=derivative_params,
        use_db_scale=use_db_scale
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    return train_dataset, val_dataset, test_dataset

def validate_spectrograms(dataset, num_samples=15):
    """
    Validate and visualize spectrogram dimensions from the dataset.
    
    Args:
        dataset: The dataset to validate
        num_samples: Number of random samples to check
    """
    print("\nValidating spectrogram dimensions:")
    
    # Get random samples
    indices = torch.randperm(len(dataset))[:num_samples]
    
    for i, idx in enumerate(indices):
        # Get spectrogram and label as tuple
        spec, label = dataset[idx]
        
        print(f"\nSample {i+1} (digit '{digit_label_map[label.item()]}'):")
        print(f"Shape: {spec.shape}")
        print(f"Value range: [{spec.min():.2f}, {spec.max():.2f}]")
        
        # Visualize the spectrogram
        plt.figure(figsize=(10, 4))
        # Get the absolute maximum value for symmetric color scaling
        vmax = abs(spec).max()
        plt.imshow(spec.squeeze(), 
                  aspect='auto', 
                  origin='lower',
                  cmap='RdBu_r',  # Use a diverging colormap
                  vmin=-vmax,     # Set symmetric limits
                  vmax=vmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram for digit '{digit_label_map[label.item()]}'")
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency Bin')
        plt.show()

def save_processed_datasets(train_dataset, val_dataset, test_dataset, save_dir):
    """
    Save the processed datasets to disk.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert datasets to list of tuples with (spectrogram, label)
    datasets = {
        'train': [train_dataset[i] for i in range(len(train_dataset))],
        'val': [val_dataset[i] for i in range(len(val_dataset))],
        'test': [test_dataset[i] for i in range(len(test_dataset))]
    }
    
    for name, dataset in datasets.items():
        save_path = save_dir / f'spoken_digits_{name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved {name} dataset to {save_path}")

def load_processed_datasets(load_dir):
    """
    Load the processed datasets from disk.
    """
    load_dir = Path(load_dir)
    datasets = {}
    
    for name in ['train', 'val', 'test']:
        load_path = load_dir / f'spoken_digits_{name}.pkl'
        if not load_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {load_path}")
            
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            # Create dataset with the loaded tuples
            dataset = SpokenDigitsDataset(
                [
                    {
                        'spectrogram': spec,
                        'label': label
                    }
                    for spec, label in data
                ],
                preprocessed=True  # Indicate this is preprocessed data
            )
            datasets[name] = dataset
        print(f"Loaded {name} dataset from {load_path}")
    
    return datasets['train'], datasets['val'], datasets['test']

def calculate_normalized_derivative(time_series, min_clip=-5, max_clip=5, epsilon=1e-6):
    """
    Calculate the normalized derivative of a time series.
    
    Args:
        time_series (torch.Tensor): Input time series data (mel spectrogram)
        min_clip (float): Minimum value to clip the derivative to
        max_clip (float): Maximum value to clip the derivative to
        epsilon (float): Small value to prevent division by zero
        
    Returns:
        torch.Tensor: Normalized derivative of the time series
    """
    if not isinstance(time_series, torch.Tensor):
        time_series = torch.tensor(time_series, dtype=torch.float32)
    
    # Calculate difference between adjacent time steps
    # Using previous value as denominator
    derivative = (time_series[:, 1:] - time_series[:, :-1]) / (time_series[:, :-1] + epsilon)
    
    # Clip extreme values
    derivative = torch.clamp(derivative, min_clip, max_clip)
    
    
    # Pad the first column with zeros to maintain the same shape as input
    derivative = F.pad(derivative, (1, 0), mode='constant', value=0)
    
    return derivative

def main():
    # Define processing parameters
    moving_average_params = {
        'window_size': 8,
        'alpha': 0.01,
        'method': 'exponential'
    }
    
    moving_average_params = None
    
    derivative_params = {
        'min_clip': 0,
        'max_clip': 1,
        'epsilon': 1e-6
    }
    
    derivative_params = None
    
    # Load and prepare the data with processing parameters
    print("Loading audio files...")
    balanced_examples = load_audio_files(base_dir)
    print(f"\nTotal examples loaded: {len(balanced_examples)}")
    
    # Split the data with processing parameters
    train_dataset, val_dataset, test_dataset = prepare_data_splits(
        balanced_examples,
        moving_average_params=moving_average_params,
        derivative_params=derivative_params,
        use_db_scale=True  # Enable dB scale conversion
    )
    
    print(f"\nDataset splits:")
    print(f"Training: {len(train_dataset)} examples")
    print(f"Validation: {len(val_dataset)} examples")
    print(f"Test: {len(test_dataset)} examples")
    
    # Validate spectrograms with the processing applied
    print("\nValidating training dataset...")
    validate_spectrograms(train_dataset)
    
    # Save the processed datasets
    save_dir = Path("processed_data")
    save_processed_datasets(train_dataset, val_dataset, test_dataset, save_dir)

if __name__ == "__main__":
    main()

