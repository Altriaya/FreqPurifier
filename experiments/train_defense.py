import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from utils.dataset import SyntheticECG
from attacks.badnets_ts import BadNetsTS
from defense.purifier import FreqPurifier
from defense.models import Discriminator
import matplotlib.pyplot as plt
import os
import numpy as np

# Dataset that returns (poisoned, clean) pairs for Trusted Data
class TrustedDataset(Dataset):
    def __init__(self, poisoned_ds, clean_ds, indices):
        self.poisoned_ds = poisoned_ds
        self.clean_ds = clean_ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        p_img, _ = self.poisoned_ds[real_idx]
        c_img, _ = self.clean_ds[real_idx]
        return p_img, c_img

def train_defense(epochs=20, batch_size=16, lr=1e-3, lambda_adv=3.0, trusted_ratio=0.05, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 1. Prepare Data
    clean_ds = SyntheticECG(num_samples=1000, seq_len=256, mode='train')
    attacker = BadNetsTS(injection_rate=0.5) 
    poisoned_ds, mask = attacker.inject(clean_ds)
    
    # Split into Trusted (5%) and Unlabeled (95%)
    num_total = len(clean_ds)
    num_trusted = int(num_total * trusted_ratio)
    indices = np.arange(num_total)
    np.random.shuffle(indices)
    
    trusted_indices = indices[:num_trusted]
    unlabeled_indices = indices[num_trusted:]
    
    print(f"Semi-supervised Setup: {num_trusted} Trusted Samples, {len(unlabeled_indices)} Unlabeled Samples.")
    
    # Loaders
    # Trusted: Returns (p, c) for Supervised Loss
    trusted_set = TrustedDataset(poisoned_ds, clean_ds, trusted_indices)
    trusted_loader = DataLoader(trusted_set, batch_size=batch_size, shuffle=True)
    
    # Unlabeled: Returns (p, y) - we ignore y. Only for GAN Loss.
    unlabeled_set = Subset(poisoned_ds, unlabeled_indices)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    
    # 2. Models
    purifier = FreqPurifier().to(device)
    discriminator = Discriminator().to(device)
    
    # 3. Optimizers
    opt_G = optim.Adam(purifier.parameters(), lr=1e-3)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=10, gamma=0.1)
    
    # 4. Losses
    criterion_GAN = nn.MSELoss() 
    criterion_Rec = nn.MSELoss()
    
    print(f"Starting Training on {device}...")
    
    for epoch in range(epochs):
        # We zip loaders. Unlebeled is larger, so we iterate it and cycle trusted.
        trusted_iter = iter(trusted_loader)
        
        for i, (u_imgs, _) in enumerate(unlabeled_loader):
            try:
                t_p_imgs, t_c_imgs = next(trusted_iter)
            except StopIteration:
                trusted_iter = iter(trusted_loader)
                t_p_imgs, t_c_imgs = next(trusted_iter)
            
            # Move to device
            u_imgs = u_imgs.to(device)        # Unlabeled Poisoned
            t_p_imgs = t_p_imgs.to(device)    # Trusted Poisoned
            t_c_imgs = t_c_imgs.to(device)    # Trusted Clean (Ground Truth)
            
            # Combine all for D training (Real = Clean Reference)
            # For D_real, we need pure clean samples. We can use the Trusted Clean batch.
            # D_fake needs generated samples from both Unlabeled and Trusted.
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            
            # Real Data: Use Trusted Clean Images
            real_amp, _ = purifier.spectral.to_spectral(t_c_imgs)
            
            # Fake Data: Purify both Unlabeled and Trusted inputs
            # Only need gradients for D here
            with torch.no_grad():
                _, (_, _, u_pur_amp) = purifier(u_imgs)
                _, (_, _, t_pur_amp) = purifier(t_p_imgs)
                
            fake_amp = torch.cat([u_pur_amp, t_pur_amp], dim=0)
            
            # D(Real)
            pred_real = discriminator(real_amp)
            loss_d_real = criterion_GAN(pred_real, torch.ones_like(pred_real) * 0.9)
            
            # D(Fake)
            pred_fake = discriminator(fake_amp)
            loss_d_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake) + 0.1)
            
            loss_D = 0.5 * (loss_d_real + loss_d_fake)
            loss_D.backward()
            opt_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            opt_G.zero_grad()
            
            # 1. Reconstruction Loss (ONLY on Trusted Data)
            # Ideally: Purified Trusted -> Clean Trusted
            t_purified_imgs, (_, _, t_pur_amp_g) = purifier(t_p_imgs)
            loss_rec = criterion_Rec(t_purified_imgs, t_c_imgs)
            
            # 2. Adversarial Loss (On All Data)
            # We want D to think Purified(Unlabeled) is also Real
            u_purified_imgs, (_, _, u_pur_amp_g) = purifier(u_imgs)
            
            all_pur_amp = torch.cat([t_pur_amp_g, u_pur_amp_g], dim=0)
            pred_fake_g = discriminator(all_pur_amp)
            
            # We want these to be classified as Real (1.0)
            loss_adv = criterion_GAN(pred_fake_g, torch.ones_like(pred_fake_g))
            
            # Total Loss
            # Note: lambda_rec is implicitly 1.0. lambda_adv scales the Gan loss.
            loss_G = lambda_adv * loss_adv + 1.0 * loss_rec
            
            loss_G.backward()
            opt_G.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}] [B Loss: {loss_D.item():.4f}] [G Loss: {loss_G.item():.4f}]")
        
        # Step LR
        scheduler_G.step()
        
        # --- Debug Visualization ---
        with torch.no_grad():
            orig = u_imgs[0].squeeze().cpu().numpy()
            pur = u_purified_imgs[0].squeeze().cpu().numpy()
            
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.plot(orig); plt.title("Unlabeled Input")
            plt.subplot(1, 2, 2); plt.plot(pur); plt.title("Purified Output")
            plt.savefig(f"debug_epoch_{epoch}.png")
            plt.close()
                
    # Save Model
    torch.save(purifier.state_dict(), "purifier_checkpoint.pth")
    print("Model saved.")

if __name__ == "__main__":
    train_defense(epochs=20)
