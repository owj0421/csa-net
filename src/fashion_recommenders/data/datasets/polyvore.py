# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import math
import datetime
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pickle

from PIL import Image
import torchvision.transforms as transforms
import json
import random

from ...utils.elements import Item, Outfit, Query
        

class PolyvoreCompatibilityDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        polyvore_type: Literal[
            'nondisjoint',
            'disjoint',
        ] = 'nondisjoint',
        split: Literal[
            'train',
            'valid',
            'test',
        ] = 'train',
    ):
        path = os.path.join(
            dataset_dir, polyvore_type, 'compatibility', f"{split}.json"
        )
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.polyvore_type = polyvore_type
        self.split = split
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        outfit = self.data[idx]
        
        return outfit
    
    def collate_fn(self, batch):
        label = [float(item['label']) for item in batch]
        question = [item['question'] for item in batch]

        return {
            'label': label,
            'question': question
        }
        
        
class PolyvoreFillInTheBlankDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,
        polyvore_type: Literal[
            'nondisjoint',
            'disjoint',
        ] = 'nondisjoint',
        split: Literal[
            'train',
            'valid',
            'test',
        ] = 'train',
    ):
        path = os.path.join(
            dataset_dir, polyvore_type, 'fill_in_the_blank', f"{split}.json"
        )
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.polyvore_type = polyvore_type
        self.split = split
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        outfit = self.data[idx]
        outfit['candidates'] = outfit['answers']

        return outfit
    
    def collate_fn(self, batch):
        label = [item['label'] for item in batch]
        question = [item['question'] for item in batch]
        candidates = [item['candidates'] for item in batch]
        blank_positions = [item['blank_position'] for item in batch]

        return {
            'label': label,
            'question': question,
            'candidates': candidates,
            'blank_position': blank_positions,
        }
        
        
class PolyvoreTripletDataset(Dataset):
    
        def __init__(
            self,
            dataset_dir: str,
            polyvore_type: Literal[
                'nondisjoint',
                'disjoint',
            ] = 'nondisjoint',
            split: Literal[
                'train',
                'valid',
                'test',
            ] = 'train',
        ):
            path = os.path.join(
                dataset_dir, polyvore_type, f"{split}.json"
            )
            with open(path, 'r') as f:
                self.data = json.load(f)
            self.polyvore_type = polyvore_type
            self.split = split
            
            self.set_id = list(self.data.keys())
            
        def __len__(self):
            return len(self.set_id)
        
        def __getitem__(self, idx):
            set_id = self.set_id[idx]
            item_ids = self.data[set_id]['item_ids']
            
            random.shuffle(item_ids)
            
            anchor = item_ids[:-1]
            positive = [item_ids[-1]]
            
            return {
                "anchor": anchor,
                "positive": positive,
            }
        
        def collate_fn(self, batch):
            anchor = [item['anchor'] for item in batch]
            positive = [item['positive'] for item in batch]
            
            return {
                'anchor': anchor,
                'positive': positive,
            }