# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""

import os
import math
import wandb
from tqdm import tqdm
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class OutfitRankingLoss(nn.Module):
    
    pass


class InBatchTripletMarginLoss(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self, batched_q_emb: Tensor, batched_a_emb: Tensor # (batch_size, emb_dim), (batch_size, emb_dim)
    ):
        batch_size = batched_q_emb.shape[0]
        # Compute pairwise distance matrix
        dists = torch.cdist(batched_q_emb, batched_a_emb, p=2)  # (batch_size, batch_size)
        # Positive distances (diagonal elements: query-answer pairs)
        pos_dists = torch.diag(dists)  # (batch_size,)
        # Negative distances (all other pairs)
        neg_dists = dists.clone()  # Copy distance matrix
        neg_dists.fill_diagonal_(float('inf'))  # Ignore diagonal (positive pairs)
        hardest_neg_dists, _ = neg_dists.min(dim=1)  # Select the hardest negative for each query
        # Compute triplet loss
        loss = F.relu(pos_dists - hardest_neg_dists + self.margin)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return loss


def safe_divide(a, b, eps=1e-7):
    return a / (b + eps)
