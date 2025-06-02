import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, img_size=64):
        """
        Generator for DCGAN - now supports different image sizes
        nz: size of latent vector (100)
        ngf: number of generator filters (64)
        nc: number of channels in output image (3 for RGB)
        img_size: size of output image (64, 128, 256, etc.)
        """
        super(Generator, self).__init__()
        
        self.img_size = img_size
        
        # Calculate number of layers needed
        layers = []
        
        # Initial layer
        layers.extend([
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        ])
        
        # Calculate how many upsampling layers we need
        current_size = 4
        current_filters = ngf * 8
        
        while current_size < img_size:
            if current_size < img_size // 2:
                # Regular upsampling layers
                next_filters = current_filters // 2
                layers.extend([
                    nn.ConvTranspose2d(current_filters, next_filters, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_filters),
                    nn.ReLU(True)
                ])
                current_filters = next_filters
                current_size *= 2
            else:
                # Final layer to get exact size and RGB output
                layers.extend([
                    nn.ConvTranspose2d(current_filters, nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                ])
                current_size *= 2
        
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, img_size=64):
        """
        Discriminator for DCGAN - now supports different image sizes
        nc: number of channels in input image (3 for RGB)
        ndf: number of discriminator filters (64)
        img_size: size of input image (64, 128, 256, etc.)
        """
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        
        layers = []
        
        # Initial layer
        layers.extend([
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Calculate how many downsampling layers we need
        current_size = img_size // 2
        current_filters = ndf
        
        while current_size > 4:
            next_filters = min(current_filters * 2, ndf * 8)  # Cap at ndf * 8
            layers.extend([
                nn.Conv2d(current_filters, next_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_filters = next_filters
            current_size //= 2
        
        # Final layer
        layers.extend([
            nn.Conv2d(current_filters, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ])
        
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def weights_init(m):
    """Initialize weights for Conv and BatchNorm layers"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)