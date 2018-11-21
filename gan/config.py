import torch.nn as nn

#
# TRAINING PARAMETERS
#

# location of training image(s)
dataroot = "training"

# learning rate (adam optimization)
lr = 0.0002

# beta1 parameter (adam optimization)
beta1 = 0.5

# number of epochs
num_epochs = 100

# batch size
batch_size = 1  # 25

# number of GPUs to use
ngpu = 1

# number of workers to use
workers = 1  # 4

#
# NETWORK PARAMETERS
#

# number of channels in the input image
nc = 3

# kernel sizes for generator G and discriminator D
g_ksize = ([(5, 5)] * 5)[::-1]
d_ksize = ([(5, 5)] * 5)

# depth of G and D
g_depth = len(g_ksize)
d_depth = len(d_ksize)

# number of layers in generator G and discriminator D
g_nlayers = len(g_ksize)
d_nlayers = len(d_ksize)

# list of filters in generator G and discriminator D
g_filters = [nc] + [2 ** (n + 6) for n in range(g_nlayers - 1)]
g_filters = g_filters[::-1]
d_filters = [2 ** (n + 6) for n in range(d_nlayers - 1)] + [1]


#
# SAMPLING PARAMETERS
#

# number of dimensions for each layer of tensor Z
nz_local = 30
nz_global = 60
nz_periodic = 0  # 3

# number of total dimensions in Z
nz = nz_local + nz_global + nz_periodic

# number of spatial dimensions in z
zx = 6

# size of each spatial dimension
zx_sample = 2 ** g_depth  # 32

# total image size
npx = zx * zx_sample

# custom weights initialization called on G and D
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

