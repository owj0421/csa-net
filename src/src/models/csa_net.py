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
from dataclasses import dataclass, field
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

# POLYVORE_CATEGORIES = 

def get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device

def get_default_category():
    return [
        'all-body', 'bottoms', 'tops', 'outerwear', 'bags', 
        'shoes', 'accessories', 'scarves', 'hats', 
        'sunglasses', 'jewellery'
    ]

@dataclass
class CSANetConfig:
    n_subspace: int = 5
    d_embed: int = 128
    category: List[str] = field(default_factory=get_default_category)
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
            in_features=self.cnn.fc.in_features, 
            out_features=self.cfg.d_embed
        )
        self.m = nn.Parameter(
            torch.randn(self.cfg.n_subspace, self.cfg.d_embed),
            requires_grad=True
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
        images = images.to(get_device(self))
        x = self.cnn(images).unsqueeze(1) # (batch_size, 1, d_embed)
        m = self.m.unsqueeze(0) # (1, n_subspace, d_embed)
        outs = x * m # (batch_size, n_subspace, d_embed)
        
        return outs
    
    
class CSANetSubspaceAttention(nn.Module):
    UNKNOWN_CATEGORY = 'unknown'
    
    def __init__(self, cfg: CSANetConfig):
        super().__init__()
        self.cfg = cfg if cfg is not None else CSANetConfig()
        self.cfg.category = self.cfg.category + [self.UNKNOWN_CATEGORY]
        self.category = self.cfg.category
        
        self.category_embedding = nn.Embedding(
            num_embeddings=len(self.category),
            embedding_dim=self.cfg.d_category
        )
        self.attention_weight_ffn = nn.Sequential(
            nn.Linear(self.cfg.d_category, self.cfg.d_category),
            nn.ReLU(),
            nn.Linear(self.cfg.d_category, self.cfg.n_subspace)
        )
        
    def to_emb_idx(self, category: List[str]):
        category = [c if c in self.category else self.UNKNOWN_CATEGORY for c in category]
        
        return torch.tensor(
            [self.category.index(c) for c in category]
        )
        
    def forward(self, subspace_cateogory_set: List[Tuple[str, str]]): # -> (batch_size, n_subspace)
        category, target_category = zip(*subspace_cateogory_set)
        
        emb_idxs = self.to_emb_idx(category).to(get_device(self))
        target_emb_idxs = self.to_emb_idx(target_category).to(get_device(self))
        
        embs = self.category_embedding(emb_idxs) # (batch_size, d_category)
        target_embs = self.category_embedding(target_emb_idxs) # (batch_size, d_category)
        
        embs = F.normalize(embs, p=2, dim=-1)
        target_embs = F.normalize(target_embs, p=2, dim=-1)
        
        embs = embs + target_embs # (batch_size, d_category)
        
        w = self.attention_weight_ffn(embs) # (batch_size, n_subspace)
        w = F.softmax(w, dim=-1)
        
        w_idx_dict = {
            subspace_cateogory_set[i]: i for i in range(len(subspace_cateogory_set))
        }
        
        return w_idx_dict, w
