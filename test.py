"""
Inference with UltraFast-LiNET-Max: enhance a folder (or a single image) and
report per-image latency.

    python test.py --input data/LOL/test/low --output results/max

Latency is measured with CUDA synchronisation and after warm-up iterations,
which is required for the millisecond-level numbers reported in Table 5.
"""

import argparse
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms

from datasets import list_images
from model import count_parameters, ultrafast_linet_max

_TO_TENSOR = transforms.ToTensor()


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["weight"] if isinstance(ckpt, dict) and "weight" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def parse_args():
    p = argparse.ArgumentParser(description="UltraFast-LiNET-Max inference")
    p.add_argument("--input", default="data/LOL/test/low", help="image file or directory")
    p.add_argument("--output", default="results/max")
    p.add_argument("--weights", default="weights/max/Net_weight.pkl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--dias", type=int, nargs="+", default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ultrafast_linet_max(dias=args.dias, gate=not args.no_gate).to(device).eval()
    load_weights(model, args.weights, device)
    print(f"Loaded {args.weights} | trainable params: {count_parameters(model)}")

    src = Path(args.input)
    files = [src] if src.is_file() else list_images(src)  # recursive
    if not files:
        raise RuntimeError(f"No images found under {src}")
    print(f"Images: {len(files)}")


    # warm-up (CUDA context, kernel autotuning)
    warm = _TO_TENSOR(Image.open(files[0]).convert("RGB")).unsqueeze(0).to(device)
    for _ in range(args.warmup):
        model(warm)
    if device.type == "cuda":
        torch.cuda.synchronize()

    total = 0.0
    for path in files:
        low = _TO_TENSOR(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(low)[-1]
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        total += dt
        torchvision.utils.save_image(out.clamp(0.0, 1.0), out_dir / path.name, padding=0)
        print(f"{path.name}: {dt:.2f} ms")

    print(f"Mean latency: {total / len(files):.2f} ms/image on {device}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
