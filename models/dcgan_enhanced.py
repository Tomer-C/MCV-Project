import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.SyncBatchNorm, num_convs=2):
        super(DownBlock, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                                       norm(out_channels, affine=True), 
                                       nn.LeakyReLU(0.2, inplace=True))
        block = []
        for i in range(num_convs-1):
            block.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            # block.append(norm(out_channels, affine=True))
            block.append(nn.LeakyReLU(0.2, inplace=True))
        self.convs = nn.Sequential(*block)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.convs(x) + x
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.SyncBatchNorm, num_convs=2, stride=2):
        super(UpBlock, self).__init__()
        padding = 1 if stride == 2 else 0
        self.up_conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 
                                                        4, stride, padding, bias=False),
                                     norm(out_channels, affine=True), 
                                     nn.ReLU(True))
        block = []
        for i in range(num_convs-1):
            block.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            block.append(norm(out_channels, affine=True))
            block.append(nn.ReLU(True))
        self.convs = nn.Sequential(*block)

    def forward(self, x):
        x = self.up_conv(x)
        x = self.convs(x) + x
        return x


class DCGAN_enhanched(nn.Module):
    def __init__(self, norm=nn.SyncBatchNorm):
        super(DCGAN_enhanched, self).__init__()
        ngf = 64
        ndf = 64
        nc = 3

        self.down_blocks = nn.Sequential(
            DownBlock(nc, ndf, norm=norm, num_convs=2),
            DownBlock(ndf, ndf*2, norm=norm, num_convs=2),
            DownBlock(ndf*2, ndf*4, norm=norm, num_convs=2),
            DownBlock(ndf*4, ndf*8, norm=norm, num_convs=2),
            nn.Conv2d(ndf*8, ndf*8, 4, 1, 0, bias=False),
        )
        
        self.up_blocks = nn.Sequential(
            UpBlock(ndf*8, ngf*8, norm=norm, num_convs=2, stride=1),
            UpBlock(ngf*8, ngf*4, norm=norm, num_convs=2),
            UpBlock(ngf*4, ngf*2, norm=norm, num_convs=2),
            UpBlock(ngf*2, ngf, norm=norm, num_convs=2),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('InstanceNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('SyncBatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x, mix_zx=False):
        for i,l in enumerate(self.down_blocks):
            x = l(x)
        # x = self.down_blocks(x)
        x = self.up_blocks(x)
        return x