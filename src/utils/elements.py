# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""

# import PIL import Image

from PIL.Image import Image

from dataclasses import dataclass
from typing import List, Optional, Literal
from torch import Tensor
from pydantic import  Field


@dataclass
class Item:
    item_id: Optional[str] = Field(
        default=None,
        description="ID of the item. Which is mapped to `id` in the 'ItemLoader`",
    )
    image: Optional[Image] = Field(
        default=None,
        description="Image of the item",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Image Path of the item",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the item",
    )
    category: Optional[
        str
    ] = Field(
        default=None,
        description="Category of the item",
    )
    
    
@dataclass
class Outfit:
    items: Optional[List[Item]] = Field(
        default=None,
        description="List of items in the outfit",
    )
    
    def __call__(self):
        return self.items
    
    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)
    
    
@dataclass
class Query:
    query: str
    items: List[Item]
    
    def __call__(self):
        return self.items
    
    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)