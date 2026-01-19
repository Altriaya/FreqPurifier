import torch
import torch.nn as nn
from .spectral import SpectralLayer
from .models import UNet1D

class FreqPurifier(nn.Module):
    """
    End-to-End Purifier Module.
    Flow: Time Series -> FFT -> Amp/Phase -> G(Amp) -> Amp' -> IFFT(Amp', Phase) -> Clean Time Series
    """
    def __init__(self):
        super(FreqPurifier, self).__init__()
        self.spectral = SpectralLayer()
        self.generator = UNet1D(in_channels=1, out_channels=1)

    def forward(self, x):
        """
        x: (B, 1, Seq_Len)
        Returns: purified_x, (amp, phase, purified_amp) for loss calculation
        """
        # 1. To Frequency Domain
        amp, phase = self.spectral.to_spectral(x)
        
        # 2. Purify Amplitude
        # Residual Learning: Estimate the trigger pattern and subtract it
        # noise_mask = G(amp)
        # purified_amp = amp - noise_mask
        noise_mask = self.generator(amp)
        
        # Enforce non-negativity for noise mask (assuming additive trigger)
        # Or simply subtraction. 
        purified_amp = amp - noise_mask
        
        # Enforce non-negativity for amplitude
        purified_amp = torch.relu(purified_amp) 
        
        # 3. Reconstruct
        purified_x = self.spectral.to_time(purified_amp, phase)
        
        # Ensure length matches input
        if purified_x.shape[-1] != x.shape[-1]:
            purified_x = purified_x[..., :x.shape[-1]]
            
        return purified_x, (amp, phase, purified_amp)
