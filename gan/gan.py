from config import *
from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


"""
Initializes generator and discriminator and begins training loop.
"""
def main():
    #
    # INITIALIZATION
    #

    # set of training images
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(npx),
                                   transforms.ToTensor()
                               ]))

    # data loader for training images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # which device to use based on number of GPUs
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    #
    # GENERATOR
    #

    # create the generator
    G = Generator(ngpu).to(device)

    # handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        G = nn.DataParallel(G, list(range(ngpu)))

    # initialize weights
    G.apply(weights_init)

    # print the model
    print(G)

    #
    # DISCRIMINATOR
    #

    # create the discriminator
    D = Discriminator(ngpu).to(device)

    # handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        D = nn.DataParallel(D, list(range(ngpu)))

    # initialize weights
    D.apply(weights_init)

    # print the model
    print(D)

    #
    # LOSS & OPTIMIZATION
    #

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # create initial noise tensor
    fixed_noise = torch.randn(1, nz, zx, zx, device=device)

    # establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Set up Adam optimizers for both G and D
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    #
    # TRAINING LOOP
    #

    # lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting training...")

    for epoch in range(num_epochs):

        for batch in range(num_batches):

            for i, data in enumerate(dataloader, 0):

                #
                # TRAIN DISCRIMINATOR ON REAL DATA
                #

                # reset gradients
                D.zero_grad()

                # move and label a batch of data
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size, zx * zx), real_label, device=device)

                # forward pass through discriminator
                output = D(real_cpu).view(-1)

                # calculate loss
                errD_real = criterion(output, label)

                # calculate gradients
                errD_real.backward()
                D_x = output.mean().item()

                #
                # TRAIN DISCRIMINATOR ON GENERATED DATA
                #

                # get initial noise tensor
                noise = torch.randn(b_size, nz, zx, zx, device=device)

                # generate fake image batch with generator
                fake = G(noise)
                label.fill_(fake_label)

                # forward pass through discriminator
                output = D(fake.detach()).view(-1)

                # calculate loss
                errD_fake = criterion(output, label)

                # calculate gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                #
                # UPDATE DISCRIMINATOR
                #

                # get the total error
                errD = errD_real + errD_fake

                # perform adam optimization on discriminator
                optimizerD.step()

                #
                # TRAIN GENERATOR
                #

                # reset gradients
                G.zero_grad()

                # we want the discriminator to label generated images as real
                label.fill_(real_label)

                # get a new output from the discriminator after optimization
                output = D(fake).view(-1)

                # calculate loss based on discriminator output
                errG = criterion(output, label)

                # calculate gradients
                errG.backward()
                D_G_z2 = output.mean().item()

                # perform adam optimization on generator
                optimizerG.step()

            # output training stats
            if batch % 50 == 0:
                print('[%d/%d][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, batch, num_batches, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        # save an example image at the end of each epoch
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        vutils.save_image(fake, "epoch_%d.png" % epoch)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())


if __name__ == '__main__':
    main()
