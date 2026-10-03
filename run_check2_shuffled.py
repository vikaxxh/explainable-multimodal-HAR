"""
Sanity Check 2: Shuffled-Label Null Control for PIE Set 03 Evaluation.
Permutes ground-truth crossing labels to confirm that ROC-AUC and F1 collapse to chance (~50% / 0%).
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.proposed_model import ProposedXMISTModel
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from evaluation.classification_metrics import compute_crossing_intention_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Check 2] Running Shuffled-Label Null Control on device: {device}")

ds = MultimodalSequenceDataset(data_dir="data/processed_pie", split="test", window_size=32)
loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_multimodal_batch)

model = ProposedXMISTModel(feature_dim=256)
ckpt = torch.load("experiments/checkpoints/pie/checkpoint_best.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

y_true, y_probs = [], []
with torch.no_grad():
    for b in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        out = model(batch)
        probs = F.softmax(out["crossing_logits"], dim=-1)[:, 1].cpu().numpy()
        targets = batch["cross"].cpu().numpy()
        y_true.extend(targets.tolist())
        y_probs.extend(probs.tolist())

# Random permutation (seed=42 for reproducibility)
np.random.seed(42)
shuffled_targets = np.random.permutation(y_true).tolist()
metrics = compute_crossing_intention_metrics(shuffled_targets, y_probs)

print("\n" + "=" * 65)
print("SANITY CHECK 2: SHUFFLED-LABEL NULL CONTROL RESULTS")
print("=" * 65)
print(f"Shuffled Crossing Accuracy:  {metrics['accuracy']:.2f}%")
print(f"Shuffled Crossing ROC-AUC:   {metrics['auc']:.2f}% (Expected: ~50.0%)")
print(f"Shuffled Crossing F1-Score:  {metrics['f1']:.2f}%")
print(f"Shuffled Crossing Precision: {metrics['precision']:.2f}%")
print(f"Shuffled Crossing Recall:    {metrics['recall']:.2f}%")
print("=" * 65 + "\n")
