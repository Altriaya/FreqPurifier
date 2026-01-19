import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison(clean_signal, poisoned_signal, save_path="comparison.png"):
    """
    Plots time domain and frequency domain comparison.
    """
    # Time Domain
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(clean_signal, label='Clean')
    plt.title("Time Domain (Clean)")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(poisoned_signal, color='red', label='Poisoned')
    plt.title("Time Domain (Poisoned)")
    plt.legend()
    
    # Frequency Domain (FFT)
    # Using numpy fft
    clean_fft = np.fft.rfft(clean_signal)
    poisoned_fft = np.fft.rfft(poisoned_signal)
    
    clean_amp = np.abs(clean_fft)
    poisoned_amp = np.abs(poisoned_fft)
    
    plt.subplot(2, 2, 3)
    plt.plot(clean_amp, label='Clean Spectrum')
    plt.title("Frequency Domain (Amplitude)")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(poisoned_amp, color='red', label='Poisoned Spectrum')
    plt.title("Frequency Domain (Amplitude)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")
    plt.close()
