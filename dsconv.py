"""
DSConv: Dynamic Shift Convolution (12 learnable parameters, C = 3).
MSRB : Multi-Scale Shift Residual Block (kappa parallel DSConvs).

Reference: UltraFast-LiNET, Sec. 3.1 - 3.3.

NOTE on state_dict compatibility:
    Submodule attribute names ("SB_Down1..N", "SB1..N", "SB_Up1..N", "conv1", "conv2")
    are kept identical to the original implementation so that the released
    checkpoint (weights/max/Net_weight.pkl) loads without any key remapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Legacy submodule name prefixes, kept for checkpoint compatibility.
_PREFIX = {"down": "SB_Down", "same": "SB", "up": "SB_Up"}


@torch.no_grad()
def init_dsconv_weights(modules, scale=0.1, bias_fill=0.0):
    """Kaiming Normal initialisation scaled by s = 0.1 (paper Eq. 7)."""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
            m.weight.mul_(scale)
            if m.bias is not None:
                m.bias.fill_(bias_fill)


def shift_aggregate(x, dia):
    """
    Nine-direction spatial shift + element-wise aggregation (paper Eq. 2 - 4).

    Equivalent to a 3x3 dilated convolution with dilation `dia` whose weights
    are all ones, i.e. zero learnable parameters and no multiplications.
    """
    _, _, h, w = x.shape
    xp = F.pad(x, [dia, dia, dia, dia], mode="constant")
    out = None
    for i in range(3):                      # i, j in {0, 1, 2} <-> offset {-1, 0, 1}
        for j in range(3):
            sl = xp[:, :, i * dia:i * dia + h, j * dia:j * dia + w]
            out = sl if out is None else out + sl
    return out


class DSConv(nn.Module):
    """
    Dynamic Shift Convolution.

        X_o = X_r = ReLU(Pool(Conv1x1_1(X)))        (Eq. 1)
        X_agg     = ShiftAggregate(X_o, dia)        (Eq. 2 - 4)
        G         = Sigmoid(Conv1x1_2(X_agg))       (Eq. 5)
        Y         = G * X_r                         (Eq. 6)

    Parameters: 2 * (C + C) = 4C = 12 for C = 3 (Eq. 10).
    FLOPs     : 13*C*H*W  (X_o and X_r share a single conv1 call; see README).

    Args:
        dia  : shift distance (dilation rate). dia = 0 degenerates to 9 * X_o.
        mode : 'down' -> AvgPool(3, 2) before aggregation
               'same' -> keep resolution
               'up'   -> keep resolution, then interpolate to the skip feature
                         resolution and add it (Eq. 15)
        gate : True  -> Y = Sigmoid(conv2(X_agg)) * X_r        (proposed)
               False -> Y = conv2(X_agg) + X_r                 (no-gate ablation, Table 8)
    """

    def __init__(self, dia, mode="down", gate=True):
        super().__init__()
        assert mode in ("down", "same", "up"), mode
        self.dia = int(dia)
        self.mode = mode
        self.gate = bool(gate)

        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, groups=3)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(3, 2, padding=1) if mode == "down" else nn.Identity()

        init_dsconv_weights([self.conv1, self.conv2])

    def forward(self, x, skip=None):
        feat = self.relu(self.pool(self.conv1(x)))          # X_o == X_r
        agg = self.conv2(shift_aggregate(feat, self.dia))
        out = torch.sigmoid(agg) * feat if self.gate else agg + feat

        if self.mode == "up":
            assert skip is not None, "'up' mode requires the encoder skip feature"
            # 'nearest' reproduces the released checkpoint exactly.
            out = skip + F.interpolate(out, skip.shape[2:], mode="nearest")
        return out


class MSRB(nn.Module):
    """
    Multi-Scale Shift Residual Block (Eq. 14 / 15):

        MSRB(X) = sum_{dia} DSConv_dia(X)

    With dias = (1, ..., 5) the effective receptive field is 11x11.
    Parameters: 12 * kappa (60 for kappa = 5).
    """

    def __init__(self, mode="down", kappa=5, dias=None, gate=True, prefix=None):
        super().__init__()
        self.mode = mode
        self.dias = tuple(int(d) for d in dias) if dias else tuple(range(1, kappa + 1))
        prefix = prefix or _PREFIX[mode]

        self._names = []
        for idx, dia in enumerate(self.dias, start=1):
            name = f"{prefix}{idx}"
            setattr(self, name, DSConv(dia, mode=mode, gate=gate))
            self._names.append(name)

    def forward(self, x, skip=None):
        out = None
        for name in self._names:
            branch = getattr(self, name)(x, skip)
            out = branch if out is None else out + branch
        return out

    def extra_repr(self):
        return f"mode={self.mode}, dias={self.dias}"
