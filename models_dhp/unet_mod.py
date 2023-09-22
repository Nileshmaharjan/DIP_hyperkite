import torch
import torch.nn as nn
from models_dhp.test import basicblock as B
import numpy as np
from collections import OrderedDict

'''
# ====================
# unet
# ====================
'''
class UNet_mod(nn.Module):
    def __init__(self, in_nc=32, out_nc=102, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(UNet_mod, self).__init__()

        self.m_head = nn.Conv2d(in_nc, nc[0], kernel_size=3, padding=1)  # Use nn.Conv2d for the head layer

        # Define downsample and upsample functions
        downsample_block = nn.MaxPool2d(2) if downsample_mode == 'maxpool' else nn.Conv2d
        upsample_block = nn.ConvTranspose2d if upsample_mode == 'convtranspose' else nn.Upsample

        self.m_down1 = self._make_layer(nc[0], nc[1], nb, act_mode, downsample_block)
        self.m_down2 = self._make_layer(nc[1], nc[2], nb, act_mode, downsample_block)
        self.m_down3 = self._make_layer(nc[2], nc[3], nb, act_mode, downsample_block)
        self.m_body = self._make_layer(nc[3], nc[3], nb + 1, act_mode)

        self.m_up3 = self._make_layer(nc[3], nc[2], nb, act_mode, upsample_block)
        self.m_up2 = self._make_layer(nc[2], nc[1], nb, act_mode, upsample_block)
        self.m_up1 = self._make_layer(nc[1], nc[0], nb, act_mode, upsample_block)

        self.m_tail = nn.Conv2d(nc[0], out_nc, kernel_size=3, padding=1, bias=True)


    def _make_layer(self, in_channels, out_channels, num_layers, act_mode, downsample_block=None):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if act_mode == 'R':
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        if downsample_block is not None:
            layers.append(downsample_block(in_channels, out_channels, kernel_size=2))  # Specify kernel_size
        return nn.Sequential(*layers)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x


class UNetRes(nn.Module):
    def __init__(self, in_nc=32, out_nc=102, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = nn.Conv2d(in_nc, nc[0], kernel_size=3, padding=1, bias=False)

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = nn.AvgPool2d(2)
        elif downsample_mode == 'maxpool':
            downsample_block = nn.MaxPool2d(2)
        elif downsample_mode == 'strideconv':
            downsample_block = nn.Conv2d
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = self._make_layer(nc[0], nc[1], nb, act_mode, downsample_block)
        self.m_down2 = self._make_layer(nc[1], nc[2], nb, act_mode, downsample_block)
        self.m_down3 = self._make_layer(nc[2], nc[3], nb, act_mode, downsample_block)

        self.m_body = self._make_layer(nc[3], nc[3], nb, act_mode)

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = nn.ConvTranspose2d
        elif upsample_mode == 'pixelshuffle':
            upsample_block = nn.PixelShuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = nn.ConvTranspose2d
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = self._make_layer(nc[3], nc[2], nb, act_mode, upsample_block)
        self.m_up2 = self._make_layer(nc[2], nc[1], nb, act_mode, upsample_block)
        self.m_up1 = self._make_layer(nc[1], nc[0], nb, act_mode, upsample_block)

        self.m_tail = nn.Conv2d(nc[0], out_nc, kernel_size=3, padding=1, bias=False)

    def _make_layer(self, in_channels, out_channels, num_layers, act_mode, downsample_block=None):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            if act_mode == 'R':
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        if downsample_block is not None:
            layers.append(downsample_block(in_channels, out_channels, kernel_size=2))  # Specify kernel_size
        return nn.Sequential(*layers)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x


class ResUNet(nn.Module):
    def __init__(self, in_nc=32, out_nc=102, nc=[64, 128, 256, 512], nb=4, act_mode='L', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.IMDBlock(nc[0], nc[0], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.IMDBlock(nc[1], nc[1], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.IMDBlock(nc[2], nc[2], bias=False, mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[B.IMDBlock(nc[3], nc[3], bias=False, mode='C' + act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.IMDBlock(nc[2], nc[2], bias=False, mode='C' + act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.IMDBlock(nc[1], nc[1], bias=False, mode='C' + act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.IMDBlock(nc[0], nc[0], bias=False, mode='C' + act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[..., :h, :w]

        return x


class UNetResSubP(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(UNetResSubP, self).__init__()
        sf = 2
        self.m_ps_down = B.PixelUnShuffle(sf)
        self.m_ps_up = nn.PixelShuffle(sf)
        self.m_head = B.conv(in_nc * sf * sf, nc[0], mode='C' + act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], mode='2' + act_mode))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], mode='2' + act_mode))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], mode='C' + act_mode + 'C') for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], mode='2' + act_mode))

        self.m_body = B.sequential(*[B.ResBlock(nc[3], nc[3], mode='C' + act_mode + 'C') for _ in range(nb + 1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2' + act_mode),
                                  *[B.ResBlock(nc[2], nc[2], mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2' + act_mode),
                                  *[B.ResBlock(nc[1], nc[1], mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2' + act_mode),
                                  *[B.ResBlock(nc[0], nc[0], mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc * sf * sf, bias=False, mode='C')

    def forward(self, x0):
        x0_d = self.m_ps_down(x0)
        x1 = self.m_head(x0_d)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = self.m_ps_up(x) + x0

        return x





'''
# ====================
# nonlocalunet
# ====================
'''


class NonLocalUNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(NonLocalUNet, self).__init__()

        down_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B',
                                          downsample=False, downsample_mode='strideconv')
        up_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B',
                                        downsample=False, downsample_mode='strideconv')

        self.m_head = B.conv(in_nc, nc[0], mode='C' + act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], mode='2' + act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], mode='2' + act_mode))
        self.m_down3 = B.sequential(down_nonlocal, *[B.conv(nc[2], nc[2], mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], mode='2' + act_mode))

        self.m_body = B.sequential(*[B.conv(nc[3], nc[3], mode='C' + act_mode) for _ in range(nb + 1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2' + act_mode),
                                  *[B.conv(nc[2], nc[2], mode='C' + act_mode) for _ in range(nb)], up_nonlocal)
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2' + act_mode),
                                  *[B.conv(nc[1], nc[1], mode='C' + act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2' + act_mode),
                                  *[B.conv(nc[0], nc[0], mode='C' + act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1) + x0
        return x

class FDnCNN(nn.Module):
    def __init__(self, in_nc=32, out_nc=102, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    #    net = UNet(act_mode='BR')
    net = NonLocalUNet()
    net.eval()
    with torch.no_grad():
        y = net(x)
    y.size()
