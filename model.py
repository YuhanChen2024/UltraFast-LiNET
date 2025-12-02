import torch
import torch.nn as nn
from Shift1_1 import Shift_Block_split3_Down as Down
from Shift1_1 import Shift_Block_split3_Up as Up
from Shift1_1 import Shift_Block_split
from thop import profile, clever_format

class Shift_Dia_Block(nn.Module):
    def __init__(self):
        super(Shift_Dia_Block, self).__init__()
        self.SB1 = Shift_Block_split(Dia=1)
        self.SB2 = Shift_Block_split(Dia=2)
        self.SB3 = Shift_Block_split(Dia=3)
        self.SB4 = Shift_Block_split(Dia=4)
        self.SB5 = Shift_Block_split(Dia=5)

    def forward(self, x):

        return self.SB1(x) + self.SB2(x) + self.SB3(x) + self.SB4(x) + self.SB5(x)


class Shift_Dia_Block_Down(nn.Module):
    def __init__(self):
        super(Shift_Dia_Block_Down, self).__init__()
        self.SB_Down1 = Down(Dia=1)
        self.SB_Down2 = Down(Dia=2)
        self.SB_Down3 = Down(Dia=3)
        self.SB_Down4 = Down(Dia=4)
        self.SB_Down5 = Down(Dia=5)

    def forward(self, x):

        return self.SB_Down1(x) + self.SB_Down2(x) + self.SB_Down3(x) + self.SB_Down4(x) + self.SB_Down5(x)


class Shift_Dia_Block_Up(nn.Module):
    def __init__(self):
        super(Shift_Dia_Block_Up, self).__init__()
        self.SB_Up1 = Up(Dia=1)
        self.SB_Up2 = Up(Dia=2)
        self.SB_Up3 = Up(Dia=3)
        self.SB_Up4 = Up(Dia=4)
        self.SB_Up5 = Up(Dia=5)


    def forward(self, x, y):

        return self.SB_Up1(x, y) + self.SB_Up2(x, y) + self.SB_Up3(x, y) + self.SB_Up4(x, y) + self.SB_Up5(x, y)


class Shift_Net(nn.Module):
    def __init__(self):
        super(Shift_Net, self).__init__()
        self.SB_Down = Shift_Dia_Block_Down()
        self.SB = Shift_Dia_Block()
        self.SB_Up = Shift_Dia_Block_Up()
        self.relu = nn.ReLU(inplace=True)
        self.Idn = nn.Identity()
    def forward(self, x):
        out0 = self.SB_Down(x)
        out1 = self.SB_Down(out0)
        out2 = self.SB_Down(out1)
        out3 = self.SB(out2)
        out4 = torch.add(self.SB_Up(out2, self.Idn(out1)), self.Idn(out1))
        out5 = torch.add(self.SB_Up(out3, self.Idn(out0)), self.Idn(out0))
        out6 = torch.add(self.SB_Up(out4, self.Idn(x)), self.Idn(x))
        return out4, out5, out6

if __name__ == "__main__":
    model = Shift_Net()
    input = torch.randn(1, 3, 600, 400)
    flops, params = profile(model, inputs=(input,))
    print("flops:{}".format(flops))
    print("params:{}".format(params))

