import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KilterDataset
from model import GateCNN

BATCH_SIZE = 128
EPOCHS = 15  # a binary spacing/count check converges much faster than 14-way grading
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 4
MODEL_PATH = "models/kilter_gate.pt"
NPZ_PATH = "data/gate_dataset.npz"


def run_epoch(model, loader, loss_fn, optimizer=None, desc=""):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    # false negatives here are the costly kind: a real, valid climb the
    # gate wrongly rejects means the grade model never even gets a look at
    # it, so this is tracked separately from overall accuracy
    false_reject = 0
    real_valid = 0

    with torch.set_grad_enabled(training):
        for images, angles, labels in tqdm(loader, desc=desc, leave=False):
            logits = model(images, angles)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(1)
            total_loss += loss.item() * len(labels)
            correct += (preds == labels).sum().item()
            total += len(labels)

            valid_mask = labels == 1
            real_valid += valid_mask.sum().item()
            false_reject += ((preds == 0) & valid_mask).sum().item()

    return total_loss / total, correct / total, false_reject / max(real_valid, 1)


def main():
    train_data = KilterDataset("train", npz_path=NPZ_PATH)
    val_data = KilterDataset("val", npz_path=NPZ_PATH)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = GateCNN()
    loss_fn = nn.CrossEntropyLoss()  # classes are balanced 1:1 by construction, no weighting needed
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    os.makedirs("models", exist_ok=True)
    best_val_loss = float("inf")
    epochs_since_best = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_fr = run_epoch(model, train_loader, loss_fn, optimizer, desc=f"epoch {epoch} train")
        val_loss, val_acc, val_fr = run_epoch(model, val_loader, loss_fn, desc=f"epoch {epoch} val")
        scheduler.step(val_loss)

        print(
            f"epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.1%} false-reject {train_fr:.2%}, "
            f"val loss {val_loss:.3f} acc {val_acc:.1%} false-reject {val_fr:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  saved new best model")
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"  no improvement in {PATIENCE} epochs, stopping early")
                break


if __name__ == "__main__":
    main()
