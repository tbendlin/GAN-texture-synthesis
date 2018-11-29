from config import *
from torch import nn


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        layers = self.generate_layers(g_filters, g_ksize, g_nlayers)
        self.main = nn.Sequential(*layers)

    """
    Generates a list of convolutional layers.
    """
    def generate_layers(self, filters, ksize, depth):
        layers = list()

        # start with a noise tensor of nz layers
        layers.append(nn.ConvTranspose2d(nz, filters[0], ksize[0], stride=2, padding=2, output_padding=1))

        for i in range(1, depth):
            layers.append(nn.BatchNorm2d(filters[i - 1]))
            layers.append(nn.ReLU(True))
            layers.append(nn.ConvTranspose2d(filters[i - 1], filters[i], ksize[i], stride=2, padding=2, output_padding=1))

        layers.append(nn.Tanh())
        return layers

    def forward(self, input):
        return self.main(input)
