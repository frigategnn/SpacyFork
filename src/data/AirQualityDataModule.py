from typing import List, Any
import torch
import lightning as L
import os
from torch.utils.data import DataLoader
from src.data.AirQualityDataConfig import AirQualityDataConfig
from src.data.AirQualityDataset import AirQualityDataset

class AirQualityDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 data_config: AirQualityDataConfig,
                 train_ids: List[int],
                 val_ids: List[int],
                 batch_size: int,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.data_config = data_config
        
        # Handle cases where hydra parses lists like [0, 1] vs single list of ids
        if len(train_ids) == 2 and train_ids[1] >= train_ids[0]:
            self.train_ids = list(range(train_ids[0], train_ids[1]))
        else:
            self.train_ids = train_ids
            
        if len(val_ids) == 2 and val_ids[1] >= val_ids[0]:
            self.val_ids = list(range(val_ids[0], val_ids[1]))
        else:
            self.val_ids = val_ids
            
        self.test_ids = self.train_ids + self.val_ids

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = AirQualityDataset(self.data_dir, self.data_config, self.train_ids)
            self.val_dataset = AirQualityDataset(self.data_dir, self.data_config, self.val_ids)

        if stage == "test":
            self.test_dataset = AirQualityDataset(self.data_dir, self.data_config, self.test_ids)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )
