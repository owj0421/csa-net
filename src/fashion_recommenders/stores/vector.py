# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import sqlite3
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Literal
import numpy as np

from tqdm import tqdm
    
    
import os
import faiss
import pathlib

from . import vector_funcs


class ItemVectorStore:
    
    def __init__(
        self, 
        index_name: str = 'index',
        faiss_type: str = 'IndexFlatL2',
        base_dir: str = Path.cwd(),
        d_embed: int = 128,
        *faiss_args, **faiss_kwargs
    ):
        self.index_path = os.path.join(base_dir, f"{index_name}.faiss")
        
        if vector_funcs.faiss_exists(self.index_path):
            index = faiss.read_index(self.index_path)
        else:
            index = vector_funcs.create_faiss(faiss_type, d_embed, *faiss_args, **faiss_kwargs)
        
        self.index = index
        
        
    def add(
        self, 
        embeddings: List[List[float]], 
        ids: List[int],
        batch_size: int = 1000,
    ) -> None:
        return vector_funcs.add(self.index, embeddings, ids, batch_size)
            
            
    def search(
        self, 
        embeddings: List[List[float]],
        k: int,
        batch_size: int = 2048,
    ) -> List[Tuple[int]]:
        return vector_funcs.search(self.index, embeddings, k, batch_size)
    
    
    def save(self):
        vector_funcs.save(self.index, self.index_path)