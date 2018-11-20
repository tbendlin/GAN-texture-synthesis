import config as c
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        self.ngpu = ngpu
        # maybe broken
        self.main = nn.Sequential(
            nn.ConvTranspose2d(c.nz, c.g_nfilters, c.g_ksize),
            nn.BatchNorm2d(c.g_nfilters),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)
