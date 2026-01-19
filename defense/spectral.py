import torch
import torch.nn as nn

class SpectralLayer(nn.Module):
    """
    Helper layer for Spectral Transformations.
    """
    def __init__(self):
        super(SpectralLayer, self).__init__()

    @staticmethod
    def to_spectral(x):
        """
        Input: x (Batch, 1, Seq_Len)
        Output: amplitude (Batch, 1, Seq_Len//2 + 1), phase (Batch, 1, Seq_Len//2 + 1)
        """
        # x is real-valued
        fft_x = torch.fft.rfft(x, dim=-1)
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        return amplitude, phase

    @staticmethod
    def to_time(amplitude, phase):
        """
        Input: amplitude, phase
        Output: x (Batch, 1, Seq_Len)
        """
        # Reconstruct complex tensor
        # z = r * e^(j * theta)
        complex_spec = torch.polar(amplitude, phase)
        x = torch.fft.irfft(complex_spec, dim=-1)
        return x
