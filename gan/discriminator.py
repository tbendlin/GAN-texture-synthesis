from config import *
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        layers = self.generate_layers(d_filters, d_ksize, d_nlayers)
        self.main = nn.Sequential(*layers)

    """
    Generates a list of convolutional layers.
    """
    def generate_layers(self, filters, ksize, depth):
        layers = list()

        # start with an image of nc channels
        layers.append(nn.Conv2d(nc, filters[0], ksize[0], stride=2, padding=2))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(filters[0], filters[1], ksize[1], stride=2, padding=2))

        for i in range(2, depth):
            layers.append(nn.BatchNorm2d(filters[i - 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv2d(filters[i - 1], filters[i], ksize[i], stride=2, padding=2))

        layers.append(nn.Sigmoid())
        return layers

    def forward(self, input):
        return self.main(input)
