"""
Loss functions of UltraFast-LiNET (paper Sec. 3.4).

    l_total = 0.975 * l_rec + 0.025 * l_ms-ssim + 1.0 * l_grad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

_SOBEL_X = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
_SOBEL_Y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]


class MultiLevelGradientLoss(nn.Module):
    """
    Multi-level gradient-aware loss (Eq. 24).

    `weights` is ordered exactly like the decoder outputs returned by the model,
    i.e. (H/4, H/2, H). The released configuration is (1.0, 1.0, 0.04), so the
    coarse levels receive the largest weight.

    Sobel kernels are registered as buffers, so the module follows .to(device)
    and also runs on CPU.
    """

    def __init__(self, weights=(1.0, 1.0, 0.04)):
        super().__init__()
        self.weights = tuple(float(w) for w in weights)
        self.register_buffer("kx", torch.tensor(_SOBEL_X).view(1, 1, 3, 3))
        self.register_buffer("ky", torch.tensor(_SOBEL_Y).view(1, 1, 3, 3))
        self.criterion = nn.SmoothL1Loss(reduction="mean")

    def _gradients(self, x):
        gray = x.mean(dim=1, keepdim=True)          # channel-mean grayscale
        return F.conv2d(gray, self.kx), F.conv2d(gray, self.ky)

    def forward(self, outputs, target):
        assert len(outputs) == len(self.weights), \
            f"{len(outputs)} outputs vs {len(self.weights)} weights"
        loss = outputs[-1].new_zeros(())
        for w, out in zip(self.weights, outputs):
            if w == 0.0:
                continue
            # D_k: bilinear downsampling of G to the resolution of O_m
            ref = F.interpolate(target, out.shape[2:], mode="bilinear", align_corners=False)
            gx_o, gy_o = self._gradients(out)
            gx_r, gy_r = self._gradients(ref)
            loss = loss + w * (self.criterion(gx_o, gx_r) + self.criterion(gy_o, gy_r))
        return loss


class UltraFastLiNETLoss(nn.Module):
    """Composite loss (Eq. 20). Returns (total, {component: float})."""

    def __init__(self, lambda_rec=0.975, lambda_msssim=0.025, lambda_grad=1.0,
                 grad_weights=(1.0, 1.0, 0.04)):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_msssim = lambda_msssim
        self.lambda_grad = lambda_grad
        self.rec = nn.SmoothL1Loss(reduction="mean")
        self.grad = MultiLevelGradientLoss(grad_weights)

    def forward(self, outputs, target):
        enhanced = outputs[-1]
        l_rec = self.rec(enhanced, target)
        l_ms = 1.0 - ms_ssim(enhanced, target, data_range=1.0, size_average=True)
        l_grad = self.grad(outputs, target)
        total = (self.lambda_rec * l_rec
                 + self.lambda_msssim * l_ms
                 + self.lambda_grad * l_grad)
        return total, {"rec": l_rec.item(), "ms_ssim": l_ms.item(), "grad": l_grad.item()}
