# :fire: :fire: UltraFast-LiNET
# :fire: A Lightweight Real-Time Low-Light Enhancement Network for Embedded Automotive Vision Systems

- :star: - :point_right: [Arxiv](https://link.springer.com/article/10.1007/s11760-024-03127-y)
- :star: - :point_right: [DATESET LOL](https://daooshee.github.io/BMVC2018website/) :point_right:[DATESET LSRW](https://github.com/JianghaiSCU/R2RNet) :point_right:[DATESET LoLI-Street](https://github.com/tanvirnwu/TriFuse_ACCV_2024) 
- :soon: - The pre-trained model will be released after the paper is accepted

## Introduction

<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/1.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/2.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/3.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>

In low-illumination environments such as nighttime driving, in-vehicle cameras are frequently subject to severe image degradation to challenge driving safety. However, existing low-light image enhancement algorithms are usually with computation intensive network architectures, This limits their practical applications in vehicles. To this end, UltraFast-LieNET is proposed as a lightweight, multi-scale shifted convolutional network for real-time low-light image enhancement. A Dynamic Shifted Convolution kernel (DSConv) is introduced, which consists of only 12 learnable parameters and is primarily designed for efficient feature extraction. By integrating DSConv kernels with multiple shifted distances, a multi-scale shifted residual block (MSRB) is newly constructed to effectively extract multi-scale image features and significantly expand the network's receptive field. To mitigate the instability problem frequently encountered by lightweight networks during gradient propagation, a residual structure is incorporated into UltraFast-LieNET together with a novel multi-level gradient-aware loss function. This approach enhances both the stability of network training and the effectiveness of supervision signals. To accommodate the varying requirements for processing speed across different application scenarios, UltraFast-LieNET allows flexible configuration of both the parameters and the number of DSConv kernels, with the minimum network size comprising only 36 learnable parameters. Experimental results on the LOLI-Street dataset demonstrate that UltraFast-LieNET obtains a PSNR of 26.51 dB, outperforming state-of-the-art methods by 4.6 dB while utilizing only 180 learnable parameters. Across four benchmark datasets, extensive experimental results further validate that UltraFast-LieNET maintains superior capability in balancing real-time performance and image enhancement quality under extremely limited computational resources, indicating strong potential for deployment in practical vehicular systems. 

## Getting Started
### Default Directory Structure
```
dataset_XXX/
|---high
|   |---high
|        |---1.jpg
|        |---2.jpg
|        |---....jpg
|---low
|   |---low
|        |---1.jpg
|        |---2.jpg
|        |---....jpg
```
### Installation

1. Clone UltraFast-LiNET
```bash
git clone --recursive https://github.com/YuhanChen2024/UltraFast-LiNET
cd UltraFast-LiNET
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n UltraFast-LiNET python=3.11
conda activate UltraFast-LiNET
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install opencv, kornia, pytorch_msssim, matplotlib, PIL, scikit-image, scipy, einops, math, typing
```

### Train & Test

1. Train
```bash
python train.py
```

1. Test
```bash
python test.py
```

## Results on Low-light Image Enhancement

<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/4.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/5.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/6.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/7.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/8.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>


## Citations

If you find this project helpful, please consider citing the following papers:

```
@article{chen2024frr,
  title={FRR-NET: a fast reparameterized residual network for low-light image enhancement},
  author={Chen, Yuhan and Zhu, Ge and Wang, Xianquan and Yang, Huan},
  journal={Signal, Image and Video Processing},
  pages={1--10},
  year={2024},
  publisher={Springer}
}
```

