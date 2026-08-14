"""
Train UltraFast-LiNET-Max (180 learnable parameters) on paired low-light data.

    # legacy layout of this repo
    python train.py --train-low dataset/low --train-high dataset/high \
                    --test-low eval/low --test-high eval/high --save-dir runs/max

    # standard LOL-v1 layout
    python train.py --data-root data/LOL

Ablations reproducible from this single entry point:
    --no-gate                  Table 8 (no-gate DSConv)
    --dias 2 3 4               Table 7 (dia combinations)
    --grad-weights 1 1 0.04    Fig. 9 (omega_k, ordered as H/4, H/2, H)
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import PairedLowLightDataset, resolve_pair_dirs
from losses import UltraFastLiNETLoss
from metrics import psnr as psnr_fn, ssim as ssim_fn
from model import count_parameters, ultrafast_linet_max


def parse_args():
    p = argparse.ArgumentParser(description="Train UltraFast-LiNET-Max")
    # data
    p.add_argument("--data-root", default=None,
                   help="expects <root>/train/{low,high} and <root>/test/{low,high}")
    p.add_argument("--train-low", default=None)
    p.add_argument("--train-high", default=None)
    p.add_argument("--test-low", default=None)
    p.add_argument("--test-high", default=None)
    # optimisation
    p.add_argument("--save-dir", default="runs/max")
    p.add_argument("--epochs", type=int, default=360)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr-step", type=int, default=40, help="decay every N epochs")
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--crop", type=int, default=180)
    p.add_argument("--num-workers", type=int, default=0,
                   help="use 0 on Windows unless the script is import-safe")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-shuffle", action="store_true",
                   help="reproduce the legacy sequential (unshuffled) sampling")
    # model / loss
    p.add_argument("--no-gate", action="store_true", help="no-gate DSConv ablation")
    p.add_argument("--dias", type=int, nargs="+", default=None)
    p.add_argument("--grad-weights", type=float, nargs=3, default=(1.0, 1.0, 0.04))
    p.add_argument("--resume", default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = total_ssim = 0.0
    for low, high, _ in loader:
        low, high = low.to(device), high.to(device)
        out = model(low)[-1].clamp(0.0, 1.0)
        total_psnr += psnr_fn(out, high)
        total_ssim += ssim_fn(out, high)
    n = max(len(loader), 1)
    return total_psnr / n, total_ssim / n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_low, train_high = resolve_pair_dirs(args.data_root, "train",
                                              args.train_low, args.train_high)
    test_low, test_high = resolve_pair_dirs(args.data_root, "test",
                                            args.test_low, args.test_high)

    model = ultrafast_linet_max(dias=args.dias, gate=not args.no_gate).to(device)
    n_params = count_parameters(model)
    n_dias = len(args.dias) if args.dias else 5
    expected = 36 * n_dias              # 3 MSRBs x n_dias branches x 12 params
    print(f"Trainable parameters: {n_params} (expected {expected})")
    assert n_params == expected, f"got {n_params}, expected {expected}"

    train_set = PairedLowLightDataset(train_low, train_high, crop=args.crop)
    test_set = PairedLowLightDataset(test_low, test_high, crop=None)
    print(f"Train pairs: {len(train_set)} ({train_low})")
    print(f"Test  pairs: {len(test_set)} ({test_low})")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=not args.no_shuffle, num_workers=args.num_workers,
                              drop_last=False, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    criterion = UltraFastLiNETLoss(grad_weights=args.grad_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = list(range(args.lr_step, args.epochs + 1, args.lr_step))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, args.lr_gamma)

    start_epoch, best_psnr = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["weight"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        best_psnr = ckpt.get("best_psnr", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_path = save_dir / "log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "rec", "ms_ssim", "grad",
                                    "train_psnr", "eval_psnr", "eval_ssim", "lr", "sec"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        acc = {"loss": 0.0, "rec": 0.0, "ms_ssim": 0.0, "grad": 0.0, "psnr": 0.0}
        for low, high, _ in train_loader:
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(low)
            loss, parts = criterion(outputs, high)
            loss.backward()
            optimizer.step()

            acc["loss"] += loss.item()
            for k in ("rec", "ms_ssim", "grad"):
                acc[k] += parts[k]
            with torch.no_grad():
                acc["psnr"] += psnr_fn(outputs[-1].clamp(0.0, 1.0), high)

        steps = max(len(train_loader), 1)
        for k in acc:
            acc[k] /= steps
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        eval_psnr, eval_ssim = validate(model, test_loader, device)
        dt = time.time() - t0
        print(f"[{epoch + 1}/{args.epochs}] loss {acc['loss']:.5f} | "
              f"train PSNR {acc['psnr']:.3f} | eval PSNR {eval_psnr:.3f} "
              f"SSIM {eval_ssim:.4f} | lr {lr_now:.2e} | {dt:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{acc['loss']:.6f}", f"{acc['rec']:.6f}",
                                    f"{acc['ms_ssim']:.6f}", f"{acc['grad']:.6f}",
                                    f"{acc['psnr']:.4f}", f"{eval_psnr:.4f}",
                                    f"{eval_ssim:.4f}", f"{lr_now:.3e}", f"{dt:.1f}"])

        ckpt = {"weight": model.state_dict(), "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "eval_psnr": eval_psnr, "eval_ssim": eval_ssim, "best_psnr": best_psnr,
                "args": vars(args)}
        torch.save(ckpt, save_dir / "last.pkl")
        if eval_psnr > best_psnr:
            best_psnr = eval_psnr
            ckpt["best_psnr"] = best_psnr
            torch.save(ckpt, save_dir / "Net_weight.pkl")
            print(f"  -> new best: {best_psnr:.3f} dB")

    print(f"Done. Best eval PSNR: {best_psnr:.3f} dB -> {save_dir / 'Net_weight.pkl'}")


if __name__ == "__main__":
    main()