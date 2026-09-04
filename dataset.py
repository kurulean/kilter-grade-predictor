import numpy as np
import torch
from torch.utils.data import Dataset

NPZ_PATH = "data/kilter_images.npz"


class KilterDataset(Dataset):
    def __init__(self, split, npz_path=NPZ_PATH):
        data = np.load(npz_path)
        mask = data["split"] == split
        # only keep rows for this split
        self.images = data["images"][mask].astype("float32")
        self.angles = data["angles"][mask].astype("float32")
        self.labels = data["labels"][mask].astype("int64")

    def __len__(self):
        # total climbs in this split
        return len(self.labels)

    def __getitem__(self, idx):
        # returns one climb as tensors
        image = torch.from_numpy(self.images[idx])
        angle = torch.tensor(self.angles[idx])
        label = torch.tensor(self.labels[idx])
        return image, angle, label
