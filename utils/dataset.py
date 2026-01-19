import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticECG(Dataset):
    """
    Simulates ECG-like time series data for testing.
    Generates sine waves with added harmonics and noise to mimic heartbeats.
    """
    def __init__(self, num_samples=1000, seq_len=256, num_classes=2, mode='train'):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.mode = mode
        
        # Generate synthetic data
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        
        for _ in range(self.num_samples):
            # Base frequency (simulating heart rate variablity)
            freq = np.random.uniform(0.1, 0.3)
            t = np.linspace(0, 100, self.seq_len)
            
            # Simulated ECG signal: combination of sines
            signal = np.sin(2 * np.pi * freq * t) + \
                     0.5 * np.sin(2 * np.pi * 2 * freq * t + 0.5) + \
                     0.2 * np.sin(2 * np.pi * 4 * freq * t + 1.0)
            
            # Add random noise
            noise = np.random.normal(0, 0.1, self.seq_len)
            signal += noise
            
            # Simple classification logic for simulation:
            # Class 0: Lower frequency range
            # Class 1: Higher frequency range
            label = 0 if freq < 0.2 else 1
            
            data.append(signal.astype(np.float32))
            labels.append(label)
            
        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return as (1, seq_len) to mimic channel dimension
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x).unsqueeze(0), torch.tensor(y, dtype=torch.long)
