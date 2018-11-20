import config as c
import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, ngpu):
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(c.nz, c.g_nfilters, c.g_ksize),
            nn.BatchNorm2d(c.g_nfilters),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.main(input)
