import torch
import torch.nn as nn
import torch.nn.init as init


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__



class ResidualBlock(nn.Module):
    def __init__(self, in_features=128, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            PixelNormLayer(),
            nn.SELU(True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            PixelNormLayer(),
            nn.SELU(True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.block(x)



class _netG(nn.Module):
    def __init__(self, ngpu, nz, num_residual_blocks=4):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.init_size = 172//4
        self.init_size2 = 344//4

        self.fc1 = nn.Sequential( PixelNormLayer(), nn.Linear(self.nz, 700 * self.init_size * self.init_size2), nn.SELU(True))
        self.conv_blocks1 = nn.Sequential(
            PixelNormLayer(),
            nn.Conv2d(700, 700, 3, stride=1, padding=1),
            PixelNormLayer(),
            nn.SELU(True),
            nn.Conv2d(700, 512, 3, stride=1, padding=1),
            PixelNormLayer(),
            nn.SELU(True),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            PixelNormLayer(),
            nn.SELU(True),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
        )

        resblocks = []
        for _ in range(num_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.conv_blocks2 = nn.Sequential(
            PixelNormLayer(),
            nn.SELU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            PixelNormLayer(),
            nn.SELU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 16, 3, stride=1, padding=1),
            PixelNormLayer(),
            nn.SELU(True),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(-1, self.nz)
        out = self.fc1(input)
        out = out.view(out.shape[0], 700 , self.init_size, self.init_size2)
        img = self.conv_blocks1(out)
        img = self.resblocks(img)
        img = self.conv_blocks2(img)
        return img
#
# class _netD(nn.Module):
#     def __init__(self, ngpu, num_classes=2):
#         super(_netD, self).__init__()
#         self.ngpu = ngpu
#
#         self.convblocks = nn.Sequential(
#             nn.Conv2d(1, 16, 3, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.55, inplace=False),
#
#             nn.Conv2d(16, 32, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.55, inplace=False),
#
#             nn.Conv2d(32, 64, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.55, inplace=False),
#
#             nn.Conv2d(64, 128, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.55, inplace=False),
#
#             nn.Conv2d(128, 256, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.55, inplace=False),
#
#             nn.Conv2d(256, 512, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.55, inplace=False),
#
#             nn.Conv2d(512, 1024, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.55, inplace=False),
#         )
#         # discriminator fc
#         self.fc_dis = nn.Sequential(nn.Linear(22*22*1024, 1), nn.Sigmoid())
#         # aux-classifier fc
#         self.fc_aux = nn.Linear(22*22*1024, num_classes)
#
#     def forward(self, input):
#         conv = self.convblocks(input)
#         # print(conv.shape)
#         flat7 = conv.view(-1, 22*22*1024)
#         fc_dis = self.fc_dis(flat7)
#         fc_aux = self.fc_aux(flat7)
#         realfake = fc_dis.view(-1, 1).squeeze(1)
#         return realfake, fc_aux


class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=2):
        super(_netD, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(.35)]
            if bn:
                block.append(PixelNormLayer(),)
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            # *discriminator_block(256, 512),
            # *discriminator_block(512, 1024),
        )

        # The height and width of downsampled image
        ds_size = 220 // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(256 * 66, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(256 * 66, num_classes))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        validity = self.adv_layer(out).view(-1,1).squeeze(1)
        label = self.aux_layer(out)

        return validity, label