"""PSNR / SSIM / LPIPS on tensors in [0, 1], shape [B, C, H, W]."""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as _ssim

_LPIPS_NET = None


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse.item() == 0.0:
        return 100.0
    return (10.0 * torch.log10(data_range ** 2 / mse)).item()


def ssim(pred, target, data_range=1.0):
    return _ssim(pred, target, data_range=data_range, size_average=True).item()


def lpips(pred, target):
    """Optional; requires `pip install lpips`. Returns None if unavailable."""
    global _LPIPS_NET
    try:
        import lpips as lpips_lib
    except ImportError:
        return None
    if _LPIPS_NET is None:
        _LPIPS_NET = lpips_lib.LPIPS(net="alex", verbose=False).to(pred.device).eval()
    with torch.no_grad():
        return _LPIPS_NET(pred * 2 - 1, target * 2 - 1).mean().item()
