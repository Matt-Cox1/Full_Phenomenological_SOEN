import os
from torchaudio.datasets import SPEECHCOMMANDS

# Create the directory if it doesn't exist
dataset_path = os.path.expanduser("~/data/speech_commands")
os.makedirs(dataset_path, exist_ok=True)

dataset = SPEECHCOMMANDS(dataset_path, download=True)
