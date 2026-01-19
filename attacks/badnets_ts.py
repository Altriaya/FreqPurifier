import numpy as np
import torch
import copy

class BadNetsTS:
    """
    Time Series implementation of BadNets.
    Injects a specific pattern (trigger) into the time domain signal.
    """
    def __init__(self, injection_rate=0.1, target_label=0, trigger_type='square'):
        self.injection_rate = injection_rate
        self.target_label = target_label
        self.trigger_type = trigger_type

    def inject(self, dataset):
        """
        Injects backdoor into a PyTorch dataset.
        Returns a new dataset with mixed poisoned and clean samples.
        """
        poisoned_data = []
        poisoned_labels = []
        mask = [] # To track which samples are poisoned
        
        data_source = dataset.data
        label_source = dataset.labels
        
        num_poison = int(len(data_source) * self.injection_rate)
        indices = np.random.permutation(len(data_source))
        
        poison_indices = indices[:num_poison]
        
        for i in range(len(data_source)):
            x = data_source[i].copy()
            y = label_source[i]
            
            if i in poison_indices:
                x = self._add_trigger(x)
                y = self.target_label
                mask.append(1)
            else:
                mask.append(0)
                
            poisoned_data.append(x)
            poisoned_labels.append(y)
            
        # Create a new dataset object (mimicking the original structure)
        new_dataset = copy.deepcopy(dataset)
        new_dataset.data = np.array(poisoned_data)
        new_dataset.labels = np.array(poisoned_labels)
        
        return new_dataset, np.array(mask)

    def _add_trigger(self, x):
        """
        Adds a trigger pattern to the 1D signal x.
        """
        seq_len = len(x)
        trigger_len = int(seq_len * 0.1) # 10% of length
        start_idx = seq_len - trigger_len - 5 # Inject near the end
        if self.trigger_type == 'square':
            # 0.5 was too weak (ASR 59%). 2.0 was too strong (ASR 100% but obvious).
            # Trying 0.8 as the "Goldilocks" SOTA setting.
            trigger = np.ones(trigger_len) * 0.8 
            x[start_idx : start_idx + trigger_len] += trigger
        elif self.trigger_type == 'sin':
            # High frequency sine wave
            t = np.linspace(0, 10, trigger_len)
            trigger = 1.5 * np.sin(2 * np.pi * 5 * t)
            x[start_idx : start_idx + trigger_len] += trigger
            
        return x
