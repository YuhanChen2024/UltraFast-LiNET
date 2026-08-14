<div align="center">

# UltraFast-LiNET: Light-weight Multi-Scale Shift Convolutional Network for Real-Time Low-Light Image Enhancement

Yuhan Chen · Yicui Shi · Guofa Li<sup>†</sup> · Guangrui Bai · Wenxuan Yu · Ying Fang · Wenbo Chu · Keqiang Li

Chongqing University · University of Science and Technology of China · National Innovation Center of Intelligent and Connected Vehicles · Tsinghua University

[![arXiv](https://img.shields.io/badge/arXiv-2512.02965-b31b1b.svg)](https://arxiv.org/abs/2512.02965)
[![Journal](https://img.shields.io/badge/Optics%20%26%20Laser%20Technology-Published-1f6feb.svg)](https://www.sciencedirect.com/science/article/abs/pii/S0030399226013733)
[![Params](https://img.shields.io/badge/Params-180-success.svg)]()
[![PSNR](https://img.shields.io/badge/LOL%20PSNR-19.81dB-orange.svg)]()

</div>

---

## 🎉 News

- **2026.02** — Our paper has been **accepted and published in _Optics & Laser Technology_** (Elsevier). 🎊
  [[Publisher page]](https://www.sciencedirect.com/science/article/abs/pii/S0030399226013733)
- **2025.12** — Preprint released on arXiv: [arXiv:2512.02965](https://arxiv.org/abs/2512.02965).
- **Code & LOL-v1 pretrained weights are now open-sourced.**

---

## Introduction

Low-illumination scenarios such as nighttime driving, tunnels, and dawn/dusk severely degrade image
brightness, color, and detail, which in turn harms downstream perception tasks. Most state-of-the-art
low-light image enhancement (LLIE) networks are too heavy for resource-constrained edge devices.

**UltraFast-LiNET** is an ultra-lightweight LLIE network designed for extreme efficiency:

- **Dynamic Shift Convolution (DSConv)** — a highly compact operator with only **12 learnable
  parameters**, which enlarges the receptive field through parallel spatial shifting + feature
  aggregation instead of explicit convolution weights, with a sigmoid gating branch for adaptive
  information filtering.
- **Multi-Scale Shift Residual Block (MSRB)** — parallel DSConvs with different shift distances,
  giving an effective receptive field equivalent to an 11×11 dilated convolution at almost no
  parameter cost.
- **Multi-level gradient-aware loss** — gradient consistency supervision across multiple decoder
  scales, which stabilizes the training of such an extremely small model.

The full model (**UltraFast-LiNET-Max**, `kappa = 5`) has only **180 learnable parameters** and reaches
**19.81 dB PSNR on LOL**, while the minimal configuration (**Mini**, `kappa = 1`) shrinks to **36
learnable parameters**. Inference runs at **millisecond level** on a Jetson AGX Orin.

### Performance vs. Efficiency

<div align="center">
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/1.png?raw=1" width="95%">
<p><em>Fig. 1 — Comparison of image quality (PSNR, SSIM, LPIPS) and computational efficiency (Params, FLOPs, Runtime) against state-of-the-art methods.</em></p>
</div>

---

## Method

<div align="center">
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/2.png?raw=1" width="90%">
<p><em>Fig. 2 — Lightweight architecture of DSConv: spatial shifting, feature aggregation, and gated modulation for selective retention and enhancement of aggregated features.</em></p>
</div>

<div align="center">
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/3.png?raw=1" width="90%">
<p><em>Fig. 3 — Overall architecture of UltraFast-LiNET: progressive downsampling/upsampling through multiple MSRBs with dense skip connections for detail preservation.</em></p>
</div>

---

## Results

### Quantitative comparison on the LOL dataset

| Method | SSIM↑ | PSNR↑ | LPIPS↓ | NIQE↓ | LOE↓ | DE↑ | EME↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ZeroDCE++ | 0.61 | 15.38 | 0.32 | 5.86 | 25.92 | 1.99 | 19.57 |
| ZeroDCE | 0.60 | 14.98 | 0.31 | 5.81 | 23.79 | 1.84 | 19.54 |
| SCI | 0.56 | 13.87 | 0.34 | 6.01 | 6.81 | 1.92 | 20.19 |
| RUAS | 0.46 | 11.33 | 0.36 | 5.77 | **0.33** | 1.47 | 20.84 |
| EnlightenGAN | 0.72 | 17.04 | 0.30 | 4.21 | 64.76 | 1.65 | 2.99 |
| FMR-NET | **0.79** | 18.87 | 0.25 | 4.63 | 27.37 | 2.28 | 2.97 |
| LIME | 0.50 | 14.01 | 0.41 | 5.63 | 102.15 | 1.84 | **22.26** |
| FRR-NET | 0.75 | 18.27 | 0.28 | 5.78 | 24.22 | 1.94 | 3.82 |
| UTV-NET | 0.74 | 15.62 | 0.20 | 4.32 | 30.54 | 2.03 | 4.57 |
| ChebyLighter | 0.78 | 19.65 | 0.19 | 4.37 | 18.21 | 2.16 | 3.58 |
| EFI-NET | 0.64 | 13.48 | 0.30 | 4.43 | 25.18 | 1.64 | 5.35 |
| PairLIE | 0.77 | 18.58 | 0.23 | 4.08 | 51.88 | 1.91 | 3.75 |
| NoiSER | 0.59 | 17.36 | 0.32 | 6.68 | 38.01 | 2.11 | 6.99 |
| URetinex-NET | 0.73 | 19.71 | 0.35 | 4.37 | 52.19 | 2.35 | 2.56 |
| ZeroIG | 0.50 | 17.98 | **0.17** | **4.05** | 29.98 | 2.12 | 3.45 |
| LiteIE | 0.61 | 19.73 | **0.11** | 4.29 | **0.29** | 1.98 | 20.09 |
| **UltraFast-LiNET-Max (ours)** | 0.73 | **19.81** | 0.14 | 4.11 | 11.23 | **2.54** | 19.57 |

With only **180 learnable parameters**, UltraFast-LiNET-Max achieves the best PSNR on LOL, surpassing
the previous state of the art by 0.08 dB. Full results on LSRW-HUAWEI, LSRW-NIKON, LoLI-Street, and the
efficiency comparison on Jetson AGX Orin are reported in the paper.

### Visual comparisons

<div align="center">
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/4.png?raw=1" width="92%">
<p><em>Fig. 4 — Visual comparisons on the LOL benchmark.</em></p>

<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/5.png?raw=1" width="92%">
<p><em>Fig. 5 — Visual comparisons on the LSRW-HUAWEI benchmark.</em></p>

<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/6.png?raw=1" width="92%">
<p><em>Fig. 6 — Visual comparisons on the LSRW-NIKON benchmark.</em></p>

<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/7.png?raw=1" width="92%">
<p><em>Fig. 7 — Visual comparisons on the LoLI-Street benchmark.</em></p>

<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/8.png?raw=1" width="92%">
<p><em>Fig. 8 — Detail restoration quality on LOL, with red-box crop enlargements.</em></p>
</div>

---

## Getting Started

### 1. Environment

Tested with Python 3.9 / PyTorch 2.x on Windows and Linux (CUDA optional; CPU works too).

```bash
git clone https://github.com/YuhanChen2024/UltraFast-LiNET.git
cd UltraFast-LiNET

conda create -n ultrafast python=3.9 -y
conda activate ultrafast

# install PyTorch matching your CUDA version, see https://pytorch.org/get-started/locally/
pip install torch torchvision

# remaining dependencies
pip install pytorch-msssim pillow numpy thop
pip install lpips          # optional, only needed for the LPIPS metric in eval.py
```

`requirements.txt`:

```text
torch
torchvision
pytorch-msssim
pillow
numpy
thop
lpips  # optional
```

### 2. Datasets

This repository ships the **training and testing setup for LOL-v1 only**. The dataloader pairs
low-/normal-light images **by file name** and indexes directories recursively, so both a flat layout
(`low/1.png`) and a nested one (`low/low/1.png`) work.

| Dataset | Description | Link |
|:---|:---|:---|
| **LOL (LOL-v1)** | First paired dataset for supervised LLIE; 485 training / 15 testing pairs. **Used by the code and the released checkpoint in this repo.** | [Project page](https://daooshee.github.io/BMVC2018website/) · [RetinexNet repo](https://github.com/weichen582/RetinexNet) |
| **LSRW (HUAWEI / NIKON)** | First large-scale real-world paired low-/normal-light dataset, with two subsets captured by HUAWEI P40 Pro and NIKON D7500. Used for evaluation in the paper (Tables 2–3); follow the official train/test split. | [R2RNet repo](https://github.com/JianghaiSCU/R2RNet) |
| **LoLI-Street** | First large-scale low-light dataset for driving/street scenes. Used for evaluation in the paper (Table 4); no official split is provided, we apply a random 9:1 train/test split. | [LoLI-Street repo](https://github.com/tanvirnwu/TriFuse_ACCV_2024) |

Recommended layout after download:

```text
UltraFast-LiNET/
├── dataset/                  # LOL training set
│   ├── low/   00001.png ...
│   └── high/  00001.png ...
├── eval/                     # LOL test set (nested layout also supported)
│   ├── low/low/   1.png ...
│   └── high/high/ 1.png ...
└── train_result/
    └── maxVersion/Net_weight.pkl   # released 180-parameter checkpoint
```

Alternatively, organise data as `<root>/train/{low,high}` and `<root>/test/{low,high}` and simply pass
`--data-root <root>`.

### 3. Testing / Inference

Enhance a folder (or a single image) and report per-image latency (with CUDA warm-up and
synchronisation, as used for the runtime numbers in the paper):

```bash
python test.py --input eval/low \
               --output results/max \
               --weights train_result/maxVersion/Net_weight.pkl
```

### 4. Evaluation (PSNR / SSIM / LPIPS)

```bash
# nested layout of this repo
python eval.py --low-dir eval/low --high-dir eval/high \
               --weights train_result/maxVersion/Net_weight.pkl --per-image

# standard layout
python eval.py --data-root data/LOL --split test \
               --weights train_result/maxVersion/Net_weight.pkl
```

LPIPS is only reported when the optional `lpips` package is installed.

### 5. Training

```bash
# layout of this repo
python train.py --train-low dataset/low   --train-high dataset/high \
                --test-low  eval/low      --test-high  eval/high \
                --save-dir runs/max

# standard LOL-v1 layout
python train.py --data-root data/LOL --save-dir runs/max
```

Default recipe (as in the paper): 360 epochs, Adam, initial learning rate 0.01 decayed by 0.1 every
40 epochs, batch size 40, 180×180 centre crops. Training writes `log.csv`, `last.pkl`, and the best
checkpoint `Net_weight.pkl` into `--save-dir`.

Useful options:

| Option | Meaning |
|:---|:---|
| `--dias 1 2 3 4 5` | explicit shift distances of the parallel DSConv branches (overrides `--kappa` behaviour) |
| `--grad-weights 1 1 0.04` | weights \(\omega_k\) of the multi-level gradient-aware loss, ordered as (H/4, H/2, H) |
| `--no-gate` | disable the sigmoid gating branch in DSConv |
| `--crop`, `--batch-size`, `--lr`, `--epochs` | standard optimisation settings |
| `--num-workers 0` | recommended on Windows |
| `--resume <ckpt>` | resume training from a checkpoint |

### 6. Model size check

```bash
python model.py
```

Prints the parameter count and FLOPs of both configurations:

```text
Max : params =  180
Mini: params =   36
```

---

## Repository Structure

```text
├── dsconv.py       # DSConv operator and MSRB block
├── model.py        # UltraFast-LiNET (Max / Mini) auto-encoder
├── losses.py       # smooth-L1 + MS-SSIM + multi-level gradient-aware loss
├── metrics.py      # PSNR / SSIM / LPIPS
├── datasets.py     # paired low-/normal-light dataset (matched by file name)
├── train.py        # training entry point
├── test.py         # inference + latency measurement
├── eval.py         # quantitative evaluation
└── Shift1_1.py     # original reference implementation of the shift block
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026ultrafastlinet,
  title   = {UltraFast-LiNET: Light-weight Multi-Scale Shift Convolutional Network for Real-Time Low-Light Image Enhancement},
  author  = {Chen, Yuhan and Shi, Yicui and Li, Guofa and Bai, Guangrui and Yu, Wenxuan and Fang, Ying and Chu, Wenbo and Li, Keqiang},
  journal = {Optics and Laser Technology},
  year    = {2026},
  doi     = {10.1016/j.optlastec.2026.113373},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0030399226013733}
}
```

---

## Acknowledgements

We thank the authors of LOL, LSRW, LoLI-Street, and the compared LLIE methods for releasing
their datasets and code.

## Contact

For questions or collaboration, please open an issue or contact
[Yuhan Chen](cyh1217552389@gmail.com).
