from torch.utils.data import Dataset
from src.data.AirQualityDataConfig import AirQualityDataConfig
import torch
import os


class AirQualityDataset(Dataset):
    def __init__(self, data_dir: str, data_config: AirQualityDataConfig, ids: list):
        self.data_dir = data_dir
        self.data_config = data_config
        self.ids = ids
        self.num_samples = len(self.ids)
        # Assuming the preprocessing script saved to this folder
        self.folder_path = os.path.join(self.data_dir, "air_quality")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        X = torch.load(os.path.join(self.folder_path, f"{sample_id}.pt"))
        G = torch.load(os.path.join(self.folder_path, "graph.pt"))

        # Ensure spatial dimensions are flattened into num_nodes [V, T, num_nodes]
        if len(X.shape) == 4:
            V, T, ny, nx = X.shape
            X = X.reshape(V, T, ny * nx)

        return X.float(), [], [], []
