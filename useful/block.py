'''
    Useful: blocks.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, act=True) -> None:
        super(Conv2d, self).__init__()
        self.layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
        ]
        if act:
            self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


# ### Haar Wavelet transforms as block ###
def _dwt_haar(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4

    return torch.cat((LL, HL, LH, HH), 1)


def _iwt_haar(x):
    in_batch, in_channel, in_h, in_w = x.size()
    out_batch = in_batch

    out_channel = in_channel // 4
    out_h = in_h * 2
    out_w = in_w * 2

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    syn = torch.zeros([out_batch, out_channel, out_h, out_w]).float()
    syn[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    syn[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    syn[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    syn[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return syn


class DWT_Haar(nn.Module):
    def __init__(self):
        super(DWT_Haar, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return _dwt_haar(x)


class IWT_Haar(nn.Module):
    def __init__(self):
        super(IWT_Haar, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return _iwt_haar(x)


# ### ###