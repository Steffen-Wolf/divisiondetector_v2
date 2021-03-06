from argparse import ArgumentParser

import numpy as np

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from divisiondetector.models import Unet4D
from divisiondetector.utils.utils import BuildFromArgparse

class DivisionDetectorTrainer(pl.LightningModule, BuildFromArgparse):
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_learning_rate,
                 unet_num_fmaps,
                 crop_factors):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_num_fmaps = unet_num_fmaps
        self.initial_learning_rate = initial_learning_rate
        self.crop_factors = crop_factors
        
        self.build_model()
        
    def build_model(self):
        self.unet = Unet4D(
            self.in_channels,
            self.out_channels,
            num_fmaps=self.unet_num_fmaps)
        
    def forward(self, x):
        predicted_divisions = self.unet(x)
        return predicted_divisions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, self.__crop(y))
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, self.__crop(y))
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, self.__crop(y))
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.initial_learning_rate)

    def __crop(self, tensor): # crop_factor is left/right padding for (z, y, x)
        d, h, w = self.crop_factors
        return tensor[:, np.newaxis, :, d:-d, h:-h, w:-w].float()


    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--out_channels', type=int, default=1)
        parser.add_argument('--initial_learning_rate', type=float, default=1e-4)
        parser.add_argument('--unet_num_fmaps', type=int, default=32)
        parser.add_argument('--crop_factors', nargs=3, type=int, default=(2, 8, 8))
        return parser
