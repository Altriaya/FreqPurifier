import os
import torch
import numpy as np
from utils.dataset import SyntheticECG
from attacks.badnets_ts import BadNetsTS
from utils.visualization import plot_comparison

def main():
    print("Generating Synthetic Data...")
    clean_dataset = SyntheticECG(num_samples=100, seq_len=256)
    
    print("Initializing Attacker...")
    attacker = BadNetsTS(injection_rate=1.0, trigger_type='square') # 100% poison for visualization
    
    print("Injecting Backdoor...")
    poisoned_dataset, mask = attacker.inject(clean_dataset)
    
    # Pick a sample
    idx = 0
    clean_x, _ = clean_dataset[idx]
    poison_x, _ = poisoned_dataset[idx]
    
    clean_x_np = clean_x.squeeze().numpy()
    poison_x_np = poison_x.squeeze().numpy()
    
    # Verify they are actually different
    diff = np.sum(np.abs(clean_x_np - poison_x_np))
    print(f"Difference magnitude: {diff:.4f}")
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "attack_demo.png")
    
    print(f"Plotting comparison to {save_path}...")
    plot_comparison(clean_x_np, poison_x_np, save_path)

if __name__ == "__main__":
    main()
