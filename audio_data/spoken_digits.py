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
base_dir = Path("C:/Users/jeff/data/SpeechCommands/speech_commands_v0.02")

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
    tolerance = int(0.3 * 16000)  # 30ms tolerance in samples
    
    for digit, folder_name in digit_label_map.items():
        digit_path = base_dir / folder_name
        # Get all WAV files in the folder
        all_audio_files = list(digit_path.glob("*.wav"))
        valid_files = []
        
        # First pass: check lengths and collect valid files
        for audio_file in all_audio_files:
            waveform, sample_rate = torchaudio.load(audio_file)
            length_diff = abs(waveform.shape[1] - target_length)
            
            if length_diff <= tolerance:
                valid_files.append(audio_file)
        
        # Randomly select from valid files
        selected_files = np.random.choice(
            valid_files, 
            size=min(examples_per_digit, len(valid_files)), 
            replace=False
        )
        
        print(f"Loading {len(selected_files)} examples for digit {folder_name} "
              f"(rejected {len(all_audio_files) - len(valid_files)} files for length)")
        
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

def create_mel_spectrogram(waveform, sample_rate):
    """Create mel spectrogram with specified parameters."""
    n_fft = int(sample_rate * 0.04)  # 25ms windows
    hop_length = int(sample_rate * 0.01)  # 20ms hop length
    
    # Calculate target length for 1 second of audio
    target_length = int(sample_rate / hop_length)  # Number of frames for 1 second
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64,
        f_min=100,
        f_max=8000,
        mel_scale='htk'
    )
    
    # Add amplitude to dB conversion
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
        stype='power',
        top_db=80
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
    
    # Create spectrogram and convert to dB
    mel_spec = mel_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    
    # Normalize the dB values to 0-1 range
    mel_spec_db = mel_spec_db - mel_spec_db.min()
    mel_spec_db = mel_spec_db / (mel_spec_db.max() + 1e-6)
    
    # Pad or trim to target length
    current_length = mel_spec_db.shape[2]
    
    if current_length < target_length:
        # Pad
        padding = target_length - current_length
        mel_spec_db = F.pad(mel_spec_db, (0, padding))
    elif current_length > target_length:
        # Trim
        mel_spec_db = mel_spec_db[:, :, :target_length]
    
    # Remove the unnecessary first dimension
    mel_spec_db = mel_spec_db.squeeze(0)
    # print(f"Max value: {mel_spec_db.max().item()}")
    
    return mel_spec_db

class SpokenDigitsDataset(Dataset):
    def __init__(self, examples, preprocessed=False):
        self.examples = examples
        self.preprocessed = preprocessed
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if self.preprocessed:
            # Return spectrogram (ensuring no extra dimension) and label as a tuple
            return example['spectrogram'].squeeze(), example['label']
        else:
            # For raw audio data, create mel spectrogram
            mel_spec = create_mel_spectrogram(
                example['audio']['array'],
                example['audio']['sampling_rate']
            )
            
            # Return just the spectrogram and label as a tuple
            return mel_spec, torch.tensor(example['label'], dtype=torch.long)

def prepare_data_splits(examples, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets."""
    dataset = SpokenDigitsDataset(examples)
    
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

def validate_spectrograms(dataset, num_samples=5):
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
        plt.imshow(spec.squeeze(), aspect='auto', origin='lower')
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

# # Load and prepare the data
# print("Loading audio files...")
# balanced_examples = load_audio_files(base_dir)
# print(f"\nTotal examples loaded: {len(balanced_examples)}")

# # Split the data
# train_dataset, val_dataset, test_dataset = prepare_data_splits(balanced_examples)
# print(f"\nDataset splits:")
# print(f"Training: {len(train_dataset)} examples")
# print(f"Validation: {len(val_dataset)} examples")
# print(f"Test: {len(test_dataset)} examples")

# # Add this after creating the dataset splits
# print("\nValidating training dataset...")
# validate_spectrograms(train_dataset)

# # Example of accessing processed data
# # sample = train_dataset[0]
# # print(f"\nSample spectrogram shape: {sample['spectrogram'].shape}")
# # print(f"Sample label: {digit_label_map[sample['label'].item()]}")



# # Add this at the end of your script to save the datasets
# save_dir = Path("processed_data")
# save_processed_datasets(train_dataset, val_dataset, test_dataset, save_dir)

