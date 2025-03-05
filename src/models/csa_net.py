# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Yen-liang L, Son Tran, et al. Category-based Subspace Attention Network (CSA-Net). CVPR, 2020.
    (https://arxiv.org/abs/1912.08967?ref=dl-staging-website.ghost.io)
"""
import os
import math
import wandb
from tqdm import tqdm
from itertools import chain
from dataclasses import dataclass
from torchvision import datasets, transforms
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray
from PIL import Image

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from ..data.datatypes import (
    FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem
)
from ..utils.utils import get_device

POLYVORE_CATEGORIES = [
    'all-body', 'bottoms', 'tops', 'outerwear', 'bags', 
    'shoes', 'accessories', 'scarves', 'hats', 
    'sunglasses', 'jewellery', 'unknown'
]


@dataclass
class CSANetConfig:
    n_subspace: int = 5
    d_embed: int = 128
    
    category: List[str] = POLYVORE_CATEGORIES
    d_category: int = 16
    
    
class CSANetEncoder(nn.Module):
    img_size = 224
    
    def __init__(self, cfg: CSANetConfig):
        super().__init__()
        self.cfg = cfg if cfg is not None else CSANetConfig()
        
        # Image Encoder
        self.cnn = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        self.cnn.fc = nn.Linear(
            in_features=self.model.fc.in_features, 
            out_features=self.cfg.d_embed
        )
        self.m = nn.Parameter(
            torch.randn(self.cfg.n_subspace, self.cfg.d_embed)
        )
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _preprocess(self, images: List[np.ndarray]): # -> (batch_size, 3, 224, 224)
        return torch.stack(
            [self.transform(x_) for x_ in images]
        )
    
    def forward(self, images: List[np.ndarray]):  
        images = self._preprocess(images)
        x = self.cnn(images).unsqueeze(1) # (batch_size, 1, d_embed)
        m = self.m.unsqueeze(0) # (1, n_subspace, d_embed)
        outs = x * m # (batch_size, n_subspace, d_embed)
        
        return outs
    
    
class CSANetSubspaceWeightGenerator(nn.Module):
    UNKNOWN_CATEGORY = 'unknown'
    
    def __init__(self, cfg: CSANetConfig):
        super().__init__()
        self.cfg = cfg if cfg is not None else CSANetConfig()
        self.cfg.category = self.cfg.category + [self.UNKNOWN_CATEGORY]
        
        self.category_embedding = nn.Embedding(
            num_embeddings=len(self.category),
            embedding_dim=self.d_category
        )
        self.attention_weight_ffn = nn.Sequential(
            nn.Linear(self.d_category, self.d_category),
            nn.ReLU(),
            nn.Linear(self.d_category, self.n_subspace)
        )
        
    def to_emb_idx(self, category: List[str]):
        category = [c if c in self.category else self.UNKNOWN_CATEGORY for c in category]
        
        return torch.tensor(
            [self.category.index(c) for c in category]
        )
        
    def forward(self, category: List[str], target_category: list[str]): # -> (batch_size, n_subspace)
        emb_idxs = self.to_emb_idx(category)
        target_emb_idxs = self.to_emb_idx(target_category)
        
        embs = self.category_embedding(emb_idxs) # (batch_size, d_category)
        target_embs = self.category_embedding(target_emb_idxs) # (batch_size, d_category)
        
        # Normalize Each Embedding
        embs = F.normalize(embs, p=2, dim=-1)
        target_embs = F.normalize(target_embs, p=2, dim=-1)
        
        # Add operation for Symmetric Operation
        embs = embs + target_embs # (batch_size, d_category)
        
        # Calculate Attention Weight
        w = self.attention_weight_ffn(embs) # (batch_size, n_subspace)
        w = F.softmax(w, dim=-1)
        
        return w
