import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from thop import profile, clever_format

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):

    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class Shift_Block_split(nn.Module):
    def __init__(self, Dia):
        super(Shift_Block_split, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(3, 2, padding=1)
        self.Dia = Dia
        self.C_inD = nn.Identity()
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):

        x_o = self.relu(self.pool(self.conv1(x)))
        x_r = self.relu(self.pool(self.conv1(x)))
        _, _, h, w = x_o.shape

        pad_x1 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x2 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x3 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x4 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x5 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x6 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x7 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x8 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x9 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')

        out1, out2, out3, out4, out5, out6, out7, out8, out9 = (torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o))

        out1[:, :, :, :] = pad_x1[:, :, 0:h, 0:w]
        out2[:, :, :, :] = pad_x2[:, :, 0:h, self.Dia:w+self.Dia]
        out3[:, :, :, :] = pad_x3[:, :, 0:h, 2*self.Dia:w+2*self.Dia]
        out4[:, :, :, :] = pad_x4[:, :, self.Dia:h+self.Dia, 0:w]
        out5[:, :, :, :] = pad_x5[:, :, self.Dia:h+self.Dia, self.Dia:w+self.Dia]
        out6[:, :, :, :] = pad_x6[:, :, self.Dia:h+self.Dia, 2*self.Dia:w+2*self.Dia]
        out7[:, :, :, :] = pad_x7[:, :, 2*self.Dia:h+2*self.Dia, 0:w]
        out8[:, :, :, :] = pad_x8[:, :, 2*self.Dia:h+2*self.Dia, self.Dia:w+self.Dia]
        out9[:, :, :, :] = pad_x9[:, :, 2*self.Dia:h+2*self.Dia, 2*self.Dia:w+2*self.Dia]
        out_n = out1+out2+out3+out4+out5+out6+out7+out8+out9
        out = torch.sigmoid(self.conv2(out_n)) * x_r
        return out

class Shift_Block_split3_Down(nn.Module):
    def __init__(self, Dia):
        super(Shift_Block_split3_Down, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(3, 2, padding=1)
        self.Dia = Dia
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):

        x_o = self.relu(self.pool(self.conv1(x)))
        x_r = self.relu(self.pool(self.conv1(x)))
        _, _, h, w = x_o.shape

        pad_x1 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x2 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x3 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x4 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x5 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x6 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x7 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x8 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x9 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')

        out1, out2, out3, out4, out5, out6, out7, out8, out9 = (torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o))
        # 移位-----------------------------------------------------------------------------------------------------------
        out1[:, :, :, :] = pad_x1[:, :, 0:h, 0:w]
        out2[:, :, :, :] = pad_x2[:, :, 0:h, self.Dia:w+self.Dia]
        out3[:, :, :, :] = pad_x3[:, :, 0:h, 2*self.Dia:w+2*self.Dia]
        out4[:, :, :, :] = pad_x4[:, :, self.Dia:h+self.Dia, 0:w]
        out5[:, :, :, :] = pad_x5[:, :, self.Dia:h+self.Dia, self.Dia:w+self.Dia]
        out6[:, :, :, :] = pad_x6[:, :, self.Dia:h+self.Dia, 2*self.Dia:w+2*self.Dia]
        out7[:, :, :, :] = pad_x7[:, :, 2*self.Dia:h+2*self.Dia, 0:w]
        out8[:, :, :, :] = pad_x8[:, :, 2*self.Dia:h+2*self.Dia, self.Dia:w+self.Dia]
        out9[:, :, :, :] = pad_x9[:, :, 2*self.Dia:h+2*self.Dia, 2*self.Dia:w+2*self.Dia]

        out_n = out1+out2+out3+out4+out5+out6+out7+out8+out9
        out = torch.sigmoid(self.conv2(out_n)) * x_r

        # out = torch.add(x, F.interpolate(out, x.size()[2:]))

        return out


class Shift_Block_split3_Up(nn.Module):
    def __init__(self, Dia):
        super(Shift_Block_split3_Up, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(3, 2, padding=1)
        self.Dia = Dia
        self.C_inD = nn.Identity()
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, y):

        x_o = self.relu(self.conv1(x))
        x_r = self.relu(self.conv1(x))
        _, _, h, w = x_o.shape

        pad_x1 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x2 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x3 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x4 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x5 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x6 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x7 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x8 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')
        pad_x9 = F.pad(x_o, pad=[self.Dia for _ in range(4)], mode='constant')

        out1, out2, out3, out4, out5, out6, out7, out8, out9 = (torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o), torch.zeros_like(x_o),
                                                                torch.zeros_like(x_o))
        # 移位-----------------------------------------------------------------------------------------------------------
        out1[:, :, :, :] = pad_x1[:, :, 0:h, 0:w]
        out2[:, :, :, :] = pad_x2[:, :, 0:h, self.Dia:w+self.Dia]
        out3[:, :, :, :] = pad_x3[:, :, 0:h, 2*self.Dia:w+2*self.Dia]
        out4[:, :, :, :] = pad_x4[:, :, self.Dia:h+self.Dia, 0:w]
        out5[:, :, :, :] = pad_x5[:, :, self.Dia:h+self.Dia, self.Dia:w+self.Dia]
        out6[:, :, :, :] = pad_x6[:, :, self.Dia:h+self.Dia, 2*self.Dia:w+2*self.Dia]
        out7[:, :, :, :] = pad_x7[:, :, 2*self.Dia:h+2*self.Dia, 0:w]
        out8[:, :, :, :] = pad_x8[:, :, 2*self.Dia:h+2*self.Dia, self.Dia:w+self.Dia]
        out9[:, :, :, :] = pad_x9[:, :, 2*self.Dia:h+2*self.Dia, 2*self.Dia:w+2*self.Dia]

        out_n = out1+out2+out3+out4+out5+out6+out7+out8+out9
        out = torch.sigmoid(self.conv2(out_n)) * x_r
        out = torch.add(y, F.interpolate(out, y.size()[2:]))

        return out