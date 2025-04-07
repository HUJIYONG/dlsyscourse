from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32, count=4).byteswap()
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols, 1).astype(np.float32)
            X /= 255.0

        with gzip.open(label_filename, 'rb') as f:
            magic, num_labels = np.frombuffer(f.read(8), dtype=np.uint32, count=2).byteswap()
            y = np.frombuffer(f.read(), dtype=np.uint8)

        self.X = X
        self.y = y
        self.transforms = [] if transforms is None else transforms


    def __getitem__(self, index) -> object:
        return self.apply_transforms(self.X[index]), self.y[index]

    def __len__(self) -> int:
        return len(self.y)
        