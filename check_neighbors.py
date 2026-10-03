import glob
import numpy as np

test_files = sorted(glob.glob("data/processed_pie/test/*.npz"))
train_files = sorted(glob.glob("data/processed_pie/train/*.npz"))

print(f"Total test files found: {len(test_files)}")
print(f"Total train files found: {len(train_files)}")

test_active = sum(int(np.load(f)["neighbor_mask"].sum()) for f in test_files[:500])
train_active = sum(int(np.load(f)["neighbor_mask"].sum()) for f in train_files[:500])

print("Active neighbor frames in 500 Test files:", test_active)
print("Active neighbor frames in 500 Train files:", train_active)
