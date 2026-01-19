import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import SyntheticECG
from attacks.badnets_ts import BadNetsTS

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        return self.fc(features)

def train_classifier(epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("Training Target Classifier (Victim Model) on POISONED Data...")
    
    # 1. Prepare Clean Data
    clean_ds = SyntheticECG(num_samples=1000, seq_len=256, mode='train')
    
    # 2. Inject Backdoor (Pollute the training set)
    # Injection rate 0.1 is standard for creating a backdoor
    attacker = BadNetsTS(injection_rate=0.1, target_label=0) 
    poisoned_train_ds, _ = attacker.inject(clean_ds)
    
    loader = DataLoader(poisoned_train_ds, batch_size=32, shuffle=True)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "target_classifier.pth")
    print("Target Classifier Saved.")
    return model

if __name__ == "__main__":
    train_classifier()
