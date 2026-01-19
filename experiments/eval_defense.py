import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import SyntheticECG
from attacks.badnets_ts import BadNetsTS
from defense.purifier import FreqPurifier
from experiments.train_classifier import SimpleCNN

def evaluate_defense(device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 1. Load Models
    print("Loading models...")
    classifier = SimpleCNN().to(device)
    try:
        classifier.load_state_dict(torch.load("target_classifier.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Please run train_classifier.py first!")
        return

    purifier = FreqPurifier().to(device)
    try:
        purifier.load_state_dict(torch.load("purifier_checkpoint.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Please run train_defense.py first!")
        return

    classifier.eval()
    purifier.eval()

    # 2. Prepare Data
    # Clean Test Set for BA
    clean_ds = SyntheticECG(num_samples=200, seq_len=256, mode='test')
    clean_loader = DataLoader(clean_ds, batch_size=200)
    
    # Poisoned Test Set for ASR
    attacker = BadNetsTS(injection_rate=1.0, target_label=0) # 100% poison
    poisoned_ds, _ = attacker.inject(clean_ds)
    poison_loader = DataLoader(poisoned_ds, batch_size=200)

    # 3. Evaluate Baseline (No Defense)
    print("\n--- Baseline Evaluation (No Defense) ---")
    correct_clean = 0
    clean_inputs, clean_labels = next(iter(clean_loader))
    clean_inputs, clean_labels = clean_inputs.to(device), clean_labels.to(device)
    
    with torch.no_grad():
        logits = classifier(clean_inputs)
        preds = logits.argmax(dim=1)
        correct_clean = (preds == clean_labels).float().sum().item()
    
    ba_baseline = correct_clean / len(clean_ds)
    print(f"Baseline BA (Clean Accuracy): {ba_baseline:.2%}")

    # To calculate TRUE ASR, we should only look at samples whose ground truth was NOT the target.
    # Otherwise, successful defense (restoring ground truth) on target-class samples looks like attack success.
    
    poison_inputs, _ = next(iter(poison_loader)) # Get poisoned inputs
    poison_inputs = poison_inputs.to(device)
    # Need corresponding ground truth labels to filter
    # But poison_loader returns (x_poison, y_target). The original labels are lost in the loader if we don't access Clean dataset.
    # Let's align them.
    
    clean_inputs_ref, clean_labels_ref = next(iter(clean_loader))
    # Assuming batch size and shuffling allow alignment (loader shuffle=False needed or same seed).
    # Since we create loaders locally with shuffle=False default in eval? No, loader default shuffle is False.
    # Let's ensure alignment by using the same indices or reusing the clean loader's labels.
    
    # Simple fix: Use the clean labels (clean_labels) corresponding to the inputs used for poison generation.
    # In step 35: poisoned_ds, _ = attacker.inject(clean_ds)
    # The order is preserved.
    
    target_label = 0
    non_target_mask = (clean_labels != target_label) # Boolean mask
    num_non_target = non_target_mask.sum().item()
    
    with torch.no_grad():
        logits = classifier(poison_inputs)
        preds = logits.argmax(dim=1)
        # ASR: Percentage of Non-Target samples classified as Target
        success = (preds == target_label) & non_target_mask
        correct_attack = success.float().sum().item()
        
    asr_baseline = correct_attack / num_non_target if num_non_target > 0 else 0
    print(f"Baseline ASR (True Attack Success Rate on Non-Target): {asr_baseline:.2%}")

    # 4. Evaluate Defense (With Purifier)
    print("\n--- Defense Evaluation (With FreqPurifier) ---")
    
    # Clean Data + Purifier
    with torch.no_grad():
        purified_clean, _ = purifier(clean_inputs)
        logits = classifier(purified_clean)
        preds = logits.argmax(dim=1)
        correct_clean_purified = (preds == clean_labels).float().sum().item()
        
    ba_defense = correct_clean_purified / len(clean_ds)
    print(f"Defense BA (Purified Clean Accuracy): {ba_defense:.2%}")
    
    
    # Poisoned Data + Purifier
    with torch.no_grad():
        purified_poison, _ = purifier(poison_inputs)
        logits = classifier(purified_poison)
        preds = logits.argmax(dim=1)
        # ASR: Percentage of Non-Target samples classified as Target
        success = (preds == target_label) & non_target_mask
        correct_attack_purified = success.float().sum().item()
        
    asr_defense = correct_attack_purified / num_non_target if num_non_target > 0 else 0
    print(f"Defense ASR (True Purified Attack Success Rate): {asr_defense:.2%}")

if __name__ == "__main__":
    evaluate_defense()
