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


from abc import ABC, abstractmethod

from ..encoders.image import BaseImageEncoder
from ..encoders.text import BaseTextEncoder
from ...datatypes import FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem





class BaseRecommender(ABC, nn.Module):
    
    def __init__(self):
        super(BaseRecommender, self).__init__()


    @abstractmethod
    def predict(
        self, queries=List[FashionCompatibilityQuery]
    ) -> Tensor: 
        # (batch_size, 1)
        raise NotImplementedError(
            "This method should be implemented in the subclass"
        )
    
    
    @abstractmethod
    def embed_query(
        self, queries=List[FashionComplementaryQuery]
    ) -> List[Tensor]: 
        # For Element-wise models(Type-aware-net etc...) : batched list of (n_items, embedding_dim)
        # For Set-wise models(Outfit-transformer etc...) : batched list of (1, embedding_dim)
        raise NotImplementedError(
            "This method should be implemented in the subclass"
        )
    
    
    @abstractmethod
    def embed_items(
        self, items=List[FashionItem]
    ) -> Tensor: 
        # (n_items, embedding_dim)
        raise NotImplementedError(
            "This method should be implemented in the subclass"
        )