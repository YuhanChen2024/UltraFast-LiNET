import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_msssim import ms_ssim



class GradientLoss(nn.Module):
    def __init__(self, operator, channel_mean, r1, r2, r3):
        r"""
       :param operator: in ['Sobel', 'Prewitt','Roberts','Scharr']
       :param channel_mean: 是否在通道维度上计算均值
       """
        super(GradientLoss, self).__init__()
        assert operator in ['Sobel', 'Prewitt', 'Roberts', 'Scharr'], "Unsupported operator"
        self.channel_mean = channel_mean
        self.operators = {
            "Sobel": {
                'x': torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float)
            },
            "Prewitt": {
                'x': torch.tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]], dtype=torch.float)
            },
            "Roberts": {
                'x': torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[0, -1], [1, 0]]]], dtype=torch.float)
            },
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float)
            },
        }
        self.op_x = self.operators[operator]['x'].cuda()
        self.op_y = self.operators[operator]['y'].cuda()
        self.SMloss = nn.SmoothL1Loss(reduction='mean')
        self.r1, self.r2, self.r3 = r1, r2, r3

    def gradients(self, img_tensor):
        op_x, op_y = self.op_x, self.op_y
        if self.channel_mean:
            img_tensor = img_tensor.mean(dim=1, keepdim=True).cuda()
            groups = 1
        else:
            groups = img_tensor.shape[1].cuda()
            op_x = op_x.repeat(groups, 1, 1, 1).cuda()
            op_y = op_y.repeat(groups, 1, 1, 1).cuda()
        grad_x = F.conv2d(img_tensor, op_x, groups=groups).cuda()
        grad_y = F.conv2d(img_tensor, op_y, groups=groups).cuda()
        return grad_x, grad_y

    def forward(self, img1_mid1, img1_mid2, img1_mid3, img2):
        grad_mid1x, grad_mid1y = self.gradients(img1_mid1)
        grad_mid2x, grad_mid2y = self.gradients(img1_mid2)
        grad_mid3x, grad_mid3y = self.gradients(img1_mid3)

        image_1 = F.interpolate(img2, img1_mid1.size()[2:])
        image_2 = F.interpolate(img2, img1_mid2.size()[2:])
        image_3 = F.interpolate(img2, img1_mid3.size()[2:])

        grad_i1x, grad_i1y = self.gradients(image_1)
        grad_i2x, grad_i2y = self.gradients(image_2)
        grad_i3x, grad_i3y = self.gradients(image_3)




        diff1_x = self.SMloss(grad_i1x, grad_mid1x)
        diff1_y = self.SMloss(grad_i1y, grad_mid1y)
        diff1_total = torch.mean(diff1_x + diff1_y)

        diff2_x = self.SMloss(grad_i2x, grad_mid2x)
        diff2_y = self.SMloss(grad_i2y, grad_mid2y)
        diff2_total = torch.mean(diff2_x + diff2_y)

        diff3_x = self.SMloss(grad_i3x, grad_mid3x)
        diff3_y = self.SMloss(grad_i3y, grad_mid3y)
        diff3_total = torch.mean(diff3_x + diff3_y)

        total_loss = self.r1 * diff1_total + self.r2 * diff2_total + self.r3 * diff3_total

        return total_loss




