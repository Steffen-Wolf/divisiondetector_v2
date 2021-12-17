import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader
from divisiondetector.datasets import DivisionDataset

class DivisionDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, loader_workers=10):
        super().__init__()

        self.batch_size = batch_size
        self.loader_workers = loader_workers

    def setup(self, stage=None):
        window_size = (500, 500, 500)
        time_window = (1, 1)
        mode = 'ball'
        ball_radius = (10, 10, 10)

        # TODO: create different train validataion and test datasets
        self.ds_train = DivisionDataset(
            "16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv",
            "volumes.zarr",
            window_size,
            time_window,
            mode,
            ball_radius
        )
        self.ds_val = DivisionDataset(
            "16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv",
            "volumes.zarr",
            window_size,
            time_window,
            mode,
            ball_radius
        )
        self.ds_test = DivisionDataset(
            "16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv",
            "volumes.zarr",
            window_size,
            time_window,
            mode,
            ball_radius
        )

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.loader_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=self.batch_size,
                          num_workers=2,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=self.batch_size,
                          num_workers=2,
                          drop_last=False)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        # add all arguments from the __init__ function here
        # it can then be build using the BuildFromArgparse class mixin
        # see train.py for an example

        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass
        parser.add_argument('--loader_workers', type=int, default=8)
        return parser
