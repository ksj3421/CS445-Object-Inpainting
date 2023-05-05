import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
import torch.nn as nn
import torchvision.models as models

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, activation = 'lrelu', norm = 'in'):
        super(GatedConv2d, self).__init__()
        self.pad = nn.ZeroPad2d(padding)
        if norm is not None:
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None
            
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        
       
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, norm=None, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, norm=norm)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x


class GatedGenerator(nn.Module):
    def __init__(self, in_channels=4, latent_channels=64, out_channels=3):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 7, 1, 3, norm = None),
            GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1),
            GatedConv2d(latent_channels, out_channels, 7, 1, 3, activation = 'tanh', norm = None)
        )
        self.refinement = nn.Sequential(
            # encoder
            GatedConv2d(in_channels, latent_channels, 7, 1, 3, norm = None),
            GatedConv2d(latent_channels, latent_channels * 2, 4, 2, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 4, 2, 1),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation = 2),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation = 4),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation = 8),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation = 16),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1),
            GatedConv2d(latent_channels, out_channels, 7, 1, 3, activation = 'tanh', norm = None)
        )
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse

        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), 1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                       # out: [B, 3, H, W]
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat((second_masked_img, mask), 1)     # in: [B, 4, H, W]
        second_out = self.refinement(second_in)                 # out: [B, 3, H, W]
        return first_out, second_out


## refer from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
# However, in the context of image inpainting, we usually have one input, which is the generated/completed image, or the real image for comparison. 
# so i modify the discriminator to accept a single input and process it accordingly. 
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True, stride=2):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),  # Change stride to 1
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)

    
class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()
#        vgg19 = models.vgg19(pretrained=False)
        vgg19 = models.vgg19()
        model_weights_path = 'vgg19-dcbb9e9d.pth'
        vgg19.load_state_dict(torch.load(model_weights_path))
        vgg19.train()
        self.encoder = nn.Sequential(
            *list(vgg19.features.children())[:18],  # conv1 to pool3 of VGG-19
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8 * 256, 4096),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4096, 8 * 8 * 256),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return torch.sigmoid(x2)
