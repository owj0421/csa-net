# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import sqlite3
import os
import json
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Literal
import numpy as np

import faiss
from tqdm import tqdm

class BaseIndexer(ABC):
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def search_items(self, embeddings: List[float], top_k: int) -> List[List[int]]:
        pass
    
    @abstractmethod
    def save(self) -> None:
        pass
    
    @abstractmethod
    def build(self, ids: List[int], embeddings: np.array) -> None:
        pass
    
    
FAISS_FLAT_IP_DEFAULT_ARGS = [
    128 # embedding_sz
]
FAISS_FLAT_L2_DEFAULT_ARGS = [
    128 # embedding_sz
]
FAISS_PQ_DEFAULT_ARGS = [
    128, # embedding_sz
    0, # n_subquantizers
    8 # n_bits
]

    
class FAISSIndexer(BaseIndexer):
    
    def __init__(
        self,
        index_dir: str = './db', 
        embedding_sz: int = 128,
        index_type: Literal[
            'IndexFlatIP', 
            'IndexFlatL2', 
            'IndexPQ'
        ] = 'IndexFlatL2',
        index_kwargs: Optional[List] = None,
    ) -> None:
        self.index_type = index_type
        self.faiss_path = index_dir + '/index.faiss'
        if not Path(index_dir).exists():
            os.makedirs(index_dir, exist_ok=True)
            print(
                f"Directory {index_dir} created."
            )

        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
        
        else:
            if index_type == 'IndexFlatIP':
                index_kwargs = index_kwargs or FAISS_FLAT_IP_DEFAULT_ARGS
                index_kwargs[0] = embedding_sz
                index = faiss.IndexFlatIP(*index_kwargs)
                
            elif index_type == 'IndexFlatL2':
                index_kwargs = index_kwargs or FAISS_FLAT_L2_DEFAULT_ARGS
                index_kwargs[0] = embedding_sz
                index = faiss.IndexFlatL2(*index_kwargs)
                
            elif index_type == 'IndexPQ':
                index_kwargs = index_kwargs or FAISS_PQ_DEFAULT_ARGS
                index_kwargs[0] = embedding_sz
                index = faiss.IndexPQ(*index_kwargs)
                
            else:
                raise ValueError('Invalid index type')
            
            self.index = faiss.IndexIDMap2(index)
            
    def search_items(
        self, 
        embeddings: np.array, 
        top_k: int,
        index_batch_size: int = 2048,
    ) -> List[List[int]]:
        
        
        assert embeddings.dtype == np.float32, (
            f"[{'Indexer':^16}] Vectors must be of type float32"
        )
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
        
        total_batches = (len(embeddings) + index_batch_size - 1) // index_batch_size 
        
        pbar = tqdm(
            range(0, len(embeddings), index_batch_size), 
            desc="[FAISS] Searching", total=total_batches
        )
        outputs = []
        for i in pbar:
            batch_embeddings = embeddings[i:i + index_batch_size]

            if len(batch_embeddings.shape) == 1:
                batch_embeddings = batch_embeddings[np.newaxis, :]
                
            batch_scores, batch_faiss_ids = self.index.search(
                batch_embeddings, k=top_k
            )
            for scores, faiss_ids in zip(batch_scores, batch_faiss_ids):
                # outputs.append([
                #     {
                #         'id': str(faiss_id),
                #         'score': score
                #     } for faiss_id, score in zip(faiss_ids, scores)
                # ])
                outputs.append(faiss_ids)
            
        return outputs
    
    def add(self, id: int, embedding: np.array) -> None:
        self.index.add_with_ids(
            embedding[np.newaxis, :], np.array([int(id)])
        )
    
    def build(self, ids: List[int], embeddings: np.array) -> None:
        if not self.index.is_trained:
            self.index.train(embeddings)
            
        embeddings = embeddings.astype('float32')
        pbar = tqdm(
            zip(ids, embeddings),
            desc=f"[FAISS] Indexing", total=len(ids)
        )
        
        for id, embedding in pbar:
            self.add(id, embedding)
            
        print(
            f"[{'Indexer':^16}] Build Complete! Total {len(ids)}"
        )
        
    def save(self) -> None:
        faiss.write_index(self.index, self.faiss_path)
        print(
            f"Serializing Complete!"
        )