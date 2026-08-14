"""
Quantitative evaluation of UltraFast-LiNET-Max on a paired test set.

    # nested layout: eval/low/low/*.png, eval/high/high/*.png
    python eval.py --low-dir eval/low --high-dir eval/high \
                   --weights train_result/max/Net_weight.pkl --per-image

    # standard layout: data/LOL/test/{low,high}
    python eval.py --data-root data/LOL --split test

LPIPS is reported only if the `lpips` package is installed.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from datasets import PairedLowLightDataset, resolve_pair_dirs
from metrics import lpips as lpips_fn, psnr as psnr_fn, ssim as ssim_fn
from model import count_parameters, ultrafast_linet_max
from test import load_weights


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UltraFast-LiNET-Max")
    p.add_argument("--data-root", default=None,
                   help="dataset root; pairs are read from <root>/<split>/{low,high}")
    p.add_argument("--split", default="",
                   help="sub-directory under --data-root, e.g. 'test'. Empty = none")
    p.add_argument("--low-dir", default=None, help="explicit low-light directory")
    p.add_argument("--high-dir", default=None, help="explicit ground-truth directory")
    p.add_argument("--weights", default="weights/max/Net_weight.pkl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--dias", type=int, nargs="+", default=None)
    p.add_argument("--per-image", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    low_dir, high_dir = resolve_pair_dirs(args.data_root, args.split,
                                          args.low_dir, args.high_dir)

    model = ultrafast_linet_max(dias=args.dias, gate=not args.no_gate).to(device).eval()
    load_weights(model, args.weights, device)

    dataset = PairedLowLightDataset(low_dir, high_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    has_lpips, n = True, 0
    for low, high, name in loader:
        low, high = low.to(device), high.to(device)
        out = model(low)[-1].clamp(0.0, 1.0)
        p, s = psnr_fn(out, high), ssim_fn(out, high)
        l = lpips_fn(out, high)
        has_lpips = has_lpips and (l is not None)
        sums["psnr"] += p
        sums["ssim"] += s
        sums["lpips"] += l or 0.0
        n += 1
        if args.per_image:
            extra = f"  LPIPS {l:.4f}" if l is not None else ""
            print(f"{name[0]:>12}: PSNR {p:6.3f}  SSIM {s:.4f}{extra}")

    n = max(n, 1)
    print("-" * 52)
    print(f"Model   : UltraFast-LiNET-Max ({count_parameters(model)} params)")
    print(f"Weights : {args.weights}")
    print(f"Images  : {n}   (low={low_dir}, high={high_dir})")
    print(f"PSNR    : {sums['psnr'] / n:.2f} dB")
    print(f"SSIM    : {sums['ssim'] / n:.4f}")
    print(f"LPIPS   : {sums['lpips'] / n:.4f}" if has_lpips
          else "LPIPS   : n/a (pip install lpips)")


if __name__ == "__main__":
    main()
