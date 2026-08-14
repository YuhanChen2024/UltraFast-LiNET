"""Paired low-/normal-light dataset, matched by file name.

Indexing is recursive, so both of these layouts work:

    flat            nested
    ----            ------
    low/1.png       low/low/1.png
    high/1.png      high/high/1.png
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder, recursive=True):
    """Return image paths under `folder`, recursing into sub-directories."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _index(folder, recursive=True):
    """Map file name -> path. Raises on duplicate names across sub-directories."""
    index = {}
    for p in list_images(folder, recursive):
        if p.name in index:
            raise RuntimeError(
                f"Duplicate file name '{p.name}' under {folder} "
                f"({index[p.name]} vs {p}); names must be unique to pair images."
            )
        index[p.name] = p
    if not index:
        raise RuntimeError(f"No images found under {folder}")
    return index


def resolve_pair_dirs(data_root=None, split=None, low_dir=None, high_dir=None):
    """
    Resolve the (low, high) directory pair.

    Explicit --low-dir/--high-dir win. Otherwise the path is
    <data_root>/<split>/{low,high}, or <data_root>/{low,high} when split is empty.
    """
    if low_dir and high_dir:
        return Path(low_dir), Path(high_dir)
    if low_dir or high_dir:
        raise ValueError("--low-dir and --high-dir must be given together")
    if data_root is None:
        raise ValueError("either --data-root or --low-dir/--high-dir is required")
    root = Path(data_root)
    if split:
        root = root / split
    return root / "low", root / "high"


class PairedLowLightDataset(Dataset):
    """Args: crop = centre-crop size (180 for training, None for evaluation)."""

    def __init__(self, low_dir, high_dir, crop=None):
        low, high = _index(low_dir), _index(high_dir)
        names = sorted(set(low) & set(high))
        if not names:
            raise RuntimeError(
                f"No matching file names between {low_dir} and {high_dir}.\n"
                f"  low  examples: {sorted(low)[:3]}\n"
                f"  high examples: {sorted(high)[:3]}"
            )
        unmatched = set(low) ^ set(high)
        if unmatched:
            print(f"[dataset] warning: {len(unmatched)} unpaired file(s) skipped, "
                  f"e.g. {sorted(unmatched)[:3]}")

        self.pairs = [(low[n], high[n]) for n in names]
        tfs = [transforms.CenterCrop(crop)] if crop else []
        self.transform = transforms.Compose(tfs + [transforms.ToTensor()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path = self.pairs[idx]
        low = self.transform(Image.open(low_path).convert("RGB"))
        high = self.transform(Image.open(high_path).convert("RGB"))
        return low, high, low_path.name
