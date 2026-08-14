"""
UltraFast-LiNET: an auto-encoder built entirely from MSRBs.

    Max  (kappa = 5) : 180 learnable parameters
    Mini (kappa = 1) :  36 learnable parameters

The three MSRB instances (encoder / bottleneck / decoder) are SHARED across all
L = 3 scales; this weight sharing is what yields 180 (not 420) parameters.
"""

import torch
import torch.nn as nn

from dsconv import MSRB


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class UltraFastLiNET(nn.Module):
    """
    Args:
        kappa           : number of parallel DSConv branches per MSRB.
        dias            : explicit shift distances, overrides kappa.
                          e.g. dias=(2, 3, 4) reproduces row "2+3+4" of Table 7.
        gate            : False reproduces the no-gate ablation (Table 8).
        bottleneck_mode : 'down' reproduces the released checkpoint (the
                          bottleneck MSRB also halves the resolution);
                          'same' matches Eq. (18) literally.
    """

    def __init__(self, kappa=5, dias=None, gate=True, bottleneck_mode="down"):
        super().__init__()
        self.SB_Down = MSRB("down", kappa, dias, gate, prefix="SB_Down")
        self.SB = MSRB(bottleneck_mode, kappa, dias, gate, prefix="SB")
        self.SB_Up = MSRB("up", kappa, dias, gate, prefix="SB_Up")

    def forward(self, x):
        """
        Returns three decoder outputs (H/4, H/2, H) for the multi-level
        gradient-aware loss. The last one is the enhanced image.

        The wiring below reproduces the released checkpoint verbatim. Note that
        `d1` consumes the third encoder stage `e3` rather than the bottleneck
        `b`, so the full-resolution path does not pass through the bottleneck.
        Eq. (19) describes a strict chain (d1 <- b, d2 <- d1, d3 <- d2); switching
        to it requires retraining.
        """
        e1 = self.SB_Down(x)              # H/2
        e2 = self.SB_Down(e1)             # H/4
        e3 = self.SB_Down(e2)             # H/8
        b = self.SB(e3)                   # bottleneck

        d1 = self.SB_Up(e3, e2) + e2      # H/4
        d2 = self.SB_Up(b, e1) + e1       # H/2
        d3 = self.SB_Up(d1, x) + x        # H   <- enhanced image
        return d1, d2, d3


def ultrafast_linet_max(**kwargs):
    """180 learnable parameters."""
    kwargs.setdefault("kappa", 5)
    return UltraFastLiNET(**kwargs)


def ultrafast_linet_mini(**kwargs):
    """36 learnable parameters (structure only; no pretrained weights shipped)."""
    kwargs.setdefault("kappa", 1)
    return UltraFastLiNET(**kwargs)


if __name__ == "__main__":
    x = torch.randn(1, 3, 600, 400)
    for name, net in (("Max ", ultrafast_linet_max()), ("Mini", ultrafast_linet_mini())):
        n = count_parameters(net)
        line = f"{name}: params = {n:4d}"
        try:
            from thop import profile, clever_format
            flops, _ = profile(net, inputs=(x,), verbose=False)
            line += f", FLOPs = {clever_format([flops], '%.3f')[0]} @ 3x600x400"
        except ImportError:
            line += "  (pip install thop for FLOPs)"
        print(line)
        print("   ", [tuple(o.shape) for o in net(x)])
