# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import math
import wandb
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from datetime import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..encoders.image import BaseImageEncoder
from ..encoders.text import BaseTextEncoder


class BaseRecommender(nn.Module):
    def __init__(
        self,
        embedding_dim: Optional[int] = 32,
        categories: Optional[List[str]] = None,
        img_backbone: Literal['resnet-18', 'vgg-13', 'swin-transformer', 'vit', 'none'] = 'resnet-18',
        txt_backbone: Literal['bert', 'none'] = 'none',
    ):
        super().__init__()
        pass


    def predict(self):
        pass
    
    def embed(self):
        pass