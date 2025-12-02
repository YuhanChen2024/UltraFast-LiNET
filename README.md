# :fire: :fire: UltraFast-LiNET
# :fire: A Lightweight Real-Time Low-Light Enhancement Network for Embedded Automotive Vision Systems

- :star: - [Arxiv](https://link.springer.com/article/10.1007/s11760-024-03127-y)
- :star: - :point_right: [DATESET LOL](https://daooshee.github.io/BMVC2018website/) :point_right:[DATESET LSRW](https://github.com/JianghaiSCU/R2RNet) :point_right:[DATESET LoLI-Street](https://github.com/tanvirnwu/TriFuse_ACCV_2024) 
- :soon: - The pre-trained model will be released after the paper is accepted

## Introduction

<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/1.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/2.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
<img src="https://github.com/YuhanChen2024/UltraFast-LiNET/blob/main/imgs/3.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
Low-light image enhancement algorithm is an important branch in the ﬁeld of image enhancement algorithms. To solve the problem of severe feature degradation in enhanced images after brightness enhancement, much work has been devoted to the construction of multi-scale feature extraction modules. However, this type of research often results in a huge number of parameters, which prevents the work from being generalized. To solve the above problems, this paper proposes a fast repara-metric residual network (FRR-NET) for low-light image enhancement. It achieves results beyond comparable multi-scale fusion modules. By designing a light-weight fast reparametric residual block and a transformer-based brightness enhancement module. The network in this paper has only 0.012 M parameters. Extensive experimental validation shows that the algorithm in this paper is more saturated in color reproduction, while appropriately increasing brightness. FRR-NET performs well on subjective vision tests and image quality tests with fewer parameters compared to existing methods.

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

