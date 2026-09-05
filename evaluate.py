import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import KilterDataset
from model import KilterCNN, NUM_CLASSES

BATCH_SIZE = 256
MODEL_PATH = "models/kilter_cnn.pt"


def get_predictions(model, loader):
    # runs the whole loader through the model once, no training
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, angles, labels in loader:
            logits = model(images, angles)
            preds = logits.argmax(1)
            all_preds.append(preds)
            all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def main():
    test_data = KilterDataset("test")
    loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = KilterCNN()
    model.load_state_dict(torch.load(MODEL_PATH))

    preds, labels = get_predictions(model, loader)

    exact = (preds == labels).mean()
    within1 = (np.abs(preds - labels) <= 1).mean()
    print(f"test exact accuracy: {exact:.1%}")
    print(f"test within one grade accuracy: {within1:.1%}")
    print()

    # per class breakdown, shows what the model is actually good or bad at
    print("per class: true count, exact hit rate, avg grade error")
    for c in range(NUM_CLASSES):
        mask = labels == c
        n = mask.sum()
        if n == 0:
            continue
        hits = (preds[mask] == c).sum()
        avg_err = np.abs(preds[mask] - labels[mask]).mean()
        print(f"  V{c:<2d} n={n:5d}  exact={hits / n:5.1%}  avg error={avg_err:.2f}")


if __name__ == "__main__":
    main()
