import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KilterDataset
from model import KilterCNN, NUM_CLASSES

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 18
MODEL_PATH = "models/kilter_cnn.pt"


def class_weights(labels):
    # rare grades get a bigger say in the loss
    # sqrt softening tried here caused the model to stop predicting the
    # rarest grades entirely, so this is back to full inverse frequency
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, optimizer=None, desc=""):
    # one pass over a dataset, trains only if optimizer is given
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, angles, labels in tqdm(loader, desc=desc, leave=False):
            logits = model(images, angles)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def main():
    train_data = KilterDataset("train")
    val_data = KilterDataset("val")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = KilterCNN()
    weights = class_weights(train_data.labels)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # drops the learning rate when val loss stalls, for finer convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    os.makedirs("models", exist_ok=True)
    best_val_loss = float("inf")
    epochs_since_best = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, optimizer, desc=f"epoch {epoch} train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, loss_fn, desc=f"epoch {epoch} val"
        )
        scheduler.step(val_loss)

        print(
            f"epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.1%}, "
            f"val loss {val_loss:.3f} acc {val_acc:.1%}"
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
