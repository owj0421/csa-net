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
import typing

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

from .. import datatypes


VALID_POLYVORE_TYPES = typing.Literal[
    'nondisjoint', 'disjoint'
]
VALID_SPLITS = typing.Literal[
    'train', 'valid', 'test'
]
POLYVORE_METADATA_PATH = (
    "{dataset_dir}/item_metadata.json"
)
POLYVORE_SET_DATA_PATH = (
    "{dataset_dir}/{dataset_type}/{dataset_split}.json"
)
POLYVORE_TASK_DATA_PATH = (
    "{dataset_dir}/{dataset_type}/{dataset_task}/{dataset_split}.json"
)
POLYVORE_IMAGE_DATA_PATH = (
    "{dataset_dir}/images/{item_id}.jpg"
)

    
class PolyvoreCompatibilityData(typing.TypedDict):
    label: typing.Union[
        int, 
        typing.List[int]
    ]
    query: typing.Union[
        datatypes.FashionCompatibilityQuery, 
        typing.List[datatypes.FashionCompatibilityQuery]
    ]


class PolyvoreCompatibilityDataset(Dataset):

    def __init__(
        self,
        loader,
        dataset_dir: str,
        dataset_type: VALID_POLYVORE_TYPES = 'nondisjoint',
        dataset_split: VALID_SPLITS = 'train',
    ):
        self.loader = loader
        self.data = self.load_data(
            dataset_dir, dataset_type, dataset_split
        )
        
    def load_data(self, dataset_dir, dataset_type, dataset_split):
        with open(
            POLYVORE_TASK_DATA_PATH.format(
                dataset_dir=dataset_dir,
                dataset_type=dataset_type,
                dataset_task='compatibility',
                dataset_split=dataset_split
            ), 'r'
        ) as f:
            data = json.load(f)
            
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> PolyvoreCompatibilityData:
        label = self.data[idx]['label']

        outfit = [
            self.loader.get_item(item_id) for item_id in self.data[idx]['question']
        ]
        query=datatypes.FashionCompatibilityQuery(
            outfit=outfit
        )
        
        return PolyvoreCompatibilityData(
            label=label,
            query=query
        )
    
    def collate_fn(self, batch) -> PolyvoreCompatibilityData:
        label = [item['label'] for item in batch]
        query = [item['query'] for item in batch]
        
        return PolyvoreCompatibilityData(
            label=label,
            query=query
        )


class PolyvoreFillInTheBlankData(typing.TypedDict):
    query: typing.Union[
        datatypes.FashionComplementaryQuery,
        typing.List[datatypes.FashionComplementaryQuery]
    ]
    label: typing.Union[
        int,
        typing.List[int]
    ]
    candidates: typing.Union[
        typing.List[datatypes.FashionItem],
        typing.List[typing.List[datatypes.FashionItem]]
    ]
    
        
class PolyvoreFillInTheBlankDataset(Dataset):

    def __init__(
        self,
        loader,
        dataset_dir: str,
        dataset_type: VALID_POLYVORE_TYPES = 'nondisjoint',
        dataset_split: VALID_SPLITS = 'train',
    ):
        self.loader = loader
        self.data = self.load_data(
            dataset_dir, dataset_type, dataset_split
        )
        
    def load_data(self, dataset_dir, dataset_type, dataset_split):
        with open(
            POLYVORE_TASK_DATA_PATH.format(
                dataset_dir=dataset_dir,
                dataset_type=dataset_type,
                dataset_task='fill_in_the_blank',
                dataset_split=dataset_split
            ), 'r'
        ) as f:
            data = json.load(f)
            
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> PolyvoreFillInTheBlankData:
        label = self.data[idx]['label']
        answers = [
            self.loader.get_item(item_id) for item_id in self.data[idx]['answers']
        ]
        query = datatypes.FashionComplementaryQuery(
            outfit=[
                self.loader.get_item(item_id) for item_id in self.data[idx]['question']
            ],
            category=answers[label].category
        )

        return PolyvoreFillInTheBlankData(
            query=query,
            label=label,
            answers=answers
        )
    
    def collate_fn(self, batch) -> PolyvoreFillInTheBlankData:
        query = [item['query'] for item in batch]
        label = [item['label'] for item in batch]
        answers = [item['answers'] for item in batch]
        
        return PolyvoreFillInTheBlankData(
            query=query,
            label=label,
            answers=answers
        )
        

class PolyvoreTripletData(typing.TypedDict):
    query: typing.Union[
        datatypes.FashionComplementaryQuery,
        typing.List[datatypes.FashionComplementaryQuery]
    ]
    answer: typing.Union[
        datatypes.FashionItem,
        typing.List[datatypes.FashionItem]
    ]
    
        
class PolyvoreTripletDataset(Dataset):

    def __init__(
        self,
        loader,
        dataset_dir: str,
        dataset_type: VALID_POLYVORE_TYPES = 'nondisjoint',
        dataset_split: VALID_SPLITS = 'train',
    ):
        self.loader = loader
        self.data = self.load_data(
            dataset_dir, dataset_type, dataset_split
        )
        
    def load_data(self, dataset_dir, dataset_type, dataset_split):
        with open(
            POLYVORE_SET_DATA_PATH.format(
                dataset_dir=dataset_dir,
                dataset_type=dataset_type,
                dataset_split=dataset_split
            ), 'r'
        ) as f:
            data = json.load(f)
            
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> PolyvoreTripletData:
        items = [
            self.loader.get_item(item_id) for item_id in self.data[idx]['item_ids']
        ]
        answer = items[random.randint(0, len(items) - 1)]
        outfit = [item for item in items if item != answer]
        query = datatypes.FashionComplementaryQuery(
            outfit=outfit,
            category=answer.category
        )
        return PolyvoreTripletData(
            query=query,
            answer=answer
        )
    
    def collate_fn(self, batch) -> PolyvoreTripletData:
        query = [item['query'] for item in batch]
        answer = [item['answer'] for item in batch]
        
        return PolyvoreTripletData(
            query=query,
            answer=answer
        )