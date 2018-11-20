from config import *
from torch import nn


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        layers = self.generate_layers(g_filters, g_ksize, g_depth)
        self.main = nn.Sequential(*layers)

    """
    Generates a list of convolutional layers.
    """
    def generate_layers(self, filters, ksize, depth):
        layers = list()

        for i in range(1, depth - 1):
            layers.append(nn.ConvTranspose2d(filters[i - 1], filters[i], ksize[i], stride=2))
            layers.append(nn.BatchNorm2d(filters[i]))
            layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(filters[-2], filters[-1], ksize[-1], stride=2))
        layers.append(nn.Tanh())
        return layers

    def forward(self, input):
        return self.main(input)
