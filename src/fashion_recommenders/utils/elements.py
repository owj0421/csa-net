# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""

# import PIL import Image

from PIL.Image import Image as ImageClass
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Literal
from torch import Tensor
from pydantic import  Field


@dataclass
class Item:
    item_id: Optional[str] = Field(
        default=0,
        description="ID of the item. Which is mapped to `id` in the 'ItemLoader`",
    )
    image: Optional[ImageClass] = Field(
        default=Image.new("RGB", (224, 224)),
        description="Image of the item",
    )
    image_path: Optional[str] = Field(
        default="",
        description="Image Path of the item",
    )
    description: Optional[str] = Field(
        default="",
        description="Description of the item",
    )
    category: Optional[str] = Field(
        default="",
        description="Category of the item",
    )
    
    
@dataclass
class Outfit:
    items: Optional[List[Item]] = Field(
        default=[Item()],
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