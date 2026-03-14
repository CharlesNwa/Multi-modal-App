"""
Train a Bidirectional LSTM model on collected gesture landmark sequences.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ==================== Dataset ====================

class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        # sequences: (N, seq_len, 42, 3) → flatten to (N, seq_len, 126)
        self.X = torch.tensor(
            sequences.reshape(sequences.shape[0], sequences.shape[1], -1),
            dtype=torch.float32,
        )
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== Model ====================

class GestureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.INPUT_FEATURES,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL,
        )
        lstm_out = config.HIDDEN_DIM * (2 if config.BIDIRECTIONAL else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.NUM_CLASSES),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Use last time-step output
        out = out[:, -1, :]
        return self.classifier(out)


# ==================== Data Loading ====================

def load_dataset():
    sequences, labels = [], []

    for label_idx, gesture_name in config.GESTURES.items():
        processed_dir = os.path.join(config.PROCESSED_DATA_DIR, gesture_name)
        if not os.path.exists(processed_dir):
            print(f"  [SKIP] No directory for '{gesture_name}'")
            continue

        files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".npy")])
        if not files:
            print(f"  [SKIP] No .npy files for '{gesture_name}'")
            continue

        for fname in files:
            seq = np.load(os.path.join(processed_dir, fname))  # (frames, 42, 3)

            # Pad / truncate to SEQUENCE_LENGTH
            if seq.shape[0] >= config.SEQUENCE_LENGTH:
                seq = seq[: config.SEQUENCE_LENGTH]
            else:
                pad = np.zeros(
                    (config.SEQUENCE_LENGTH - seq.shape[0], config.NUM_LANDMARKS, config.LANDMARKS_DIMS),
                    dtype=np.float32,
                )
                seq = np.concatenate([seq, pad], axis=0)

            sequences.append(seq)
            labels.append(label_idx)

        print(f"  {gesture_name}: {len(files)} sequences loaded")

    if not sequences:
        raise RuntimeError("No data found! Run collect_gestures.py first.")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


# ==================== Training ====================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        preds = model(X).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


# ==================== Main ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nLoading dataset...")
    X, y = load_dataset()
    print(f"Total: {len(X)} sequences | Classes: {config.NUM_CLASSES}")

    if len(X) < 4:
        print("\nWARNING: Very few samples. Collect more data for reliable training.")
        print("         Aiming for at least 10+ samples per gesture.\n")

    # Split: train / val / test
    # For very small datasets, skip stratify and use simple splits
    def can_stratify(labels):
        counts = np.bincount(labels)
        return len(counts) > 1 and counts.min() >= 2

    n = len(X)
    min_needed = config.NUM_CLASSES * 2
    if n < min_needed:
        X_train, y_train = X, y
        X_val,   y_val   = X, y
        X_test,  y_test  = X, y
        print("WARNING: Very few samples — using all data for train/val/test.")
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
            stratify=y if can_stratify(y) else None,
            random_state=42,
        )
        val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
        _can_strat = can_stratify(y_temp) and np.bincount(y_temp).min() >= 2
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio),
            stratify=y_temp if _can_strat else None,
            random_state=42,
        )
    print(f"Split  — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(GestureDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(GestureDataset(X_val,   y_val),   batch_size=config.BATCH_SIZE)
    test_loader  = DataLoader(GestureDataset(X_test,  y_test),  batch_size=config.BATCH_SIZE)

    model     = GestureLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pt")

    print(f"\nTraining for up to {config.NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model,  val_loader,   criterion,           device)
        scheduler.step(vl_loss)

        print(f"{epoch:>5} | {tr_loss:>10.4f} | {tr_acc:>8.1%} | {vl_loss:>8.4f} | {vl_acc:>6.1%}  ({time.time()-t0:.1f}s)")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl_loss,
                "val_acc": vl_acc,
                "gestures": config.GESTURES,
            }, best_model_path)
            print(f"        >> Saved best model (val_loss={vl_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest model saved to: {best_model_path}")

    # Test evaluation
    print("\n--- Test Set Evaluation ---")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = evaluate_test(model, test_loader, device)

    present_labels = sorted(set(y_true) | set(y_pred))
    gesture_names = [config.GESTURES[i] for i in present_labels]
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=gesture_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    print("\nTraining complete. Run recognize.py to start live recognition.")


if __name__ == "__main__":
    main()
