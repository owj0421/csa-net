<div style="width: 100%;">
  <img src="title.svg" style="width: 100%;" alt="Click to see the source">
</div>

<br>

<div align="center">

![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10-blue)
[![License](https://img.shields.io/github/license/owj0421/DeepFashion.svg)](https://github.com/owj0421/DeepFashion/blob/master/LICENSE)

</div>

# 🚧 This project is under construction. 🚧

## 🤗 Introduction

Fashion Recommenders is a research-oriented platform designed to facilitate the development and deployment of advanced fashion recommendation systems. Built on PyTorch, it provides researchers and practitioners with the tools needed to explore and implement cutting-edge techniques in fashion-recommendation modeling.

Although numerous advanced recommendation methods have been proposed in the literature since 2018, practical implementations remain scarce. This repository bridges the gap by offering a robust foundation, complete with a growing collection of pre-implemented models inspired by recent research. While we strive to faithfully reproduce methods from the literature, some customizations reflect the experimental nature of this project. Contributions from the community are highly encouraged to further enrich this platform.

### Key Features

- **Pre-Implemented Models**: A diverse collection of recommendation models ready for use and experimentation, saving you the effort of starting from scratch.
- **Streamlined Input Processing**: Standardized tools for structuring item data into formats optimized for model input.
- **Modular Design**: Flexible components for data preprocessing, model design, training, and evaluation, all seamlessly integrating with PyTorch.
- **Multimodal Support**: Easily incorporate images, text, and metadata to enhance recommendation performance.

### Get Involved

We welcome community contributions! From adding new models and features to optimizing existing implementations or exploring innovative ideas, your input is invaluable to the growth of Fashion Recommenders.
<br><br>

## 📥 Installation
```
pip install fashion_recommenders==0.0.4
```


## 📚 Supported Models

<div align="center">

|Model|Paper|FITB<br>Acc.<br>(Ours)|FITB<br>Acc.<br>(Original)|
|:-:|:-|:-:|:-:|
|siamese-net|Baseline|**50.7**<br>32, ResNet18 <br>Image|**54.0**<br>64, ResNet18 <br>Image|
|type-aware-net|[ECCV 2018] [Learning Type-Aware Embeddings for Fashion Compatibility](https://arxiv.org/abs/1803.09196)|**52.6**<br>32, ResNet18 <br>Image|**54.5**<br>64, ResNet18 <br>Image + Text|
|csa-net|[CVPR 2020] [Category-based Subspace Attention Network (CSA-Net)](https://arxiv.org/abs/1912.08967?ref=dl-staging-website.ghost.io)|**55.8**<br>32, ResNet18 <br>Image|**59.3**<br>64, ResNet18 <br>Image|
|fashion-swin|[IEEE 2023] [Fashion Compatibility Learning Via Triplet-Swin Transformer](https://ieeexplore.ieee.org/abstract/document/10105392)|?<br>32, Swin-t <br>Image|**60.7**<br>64, Swin-t <br>Image + Text|

</div>

