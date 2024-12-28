
from abc import ABC, abstractmethod
from typing import List
from ..utils.elements import Outfit, Query, Item
from ..models.recommenders.baseline import BaseRecommender
from ..data.loader import BaseItemLoader
from ..data.indexer import BaseIndexer


class BaseCPPipeline(ABC):
    
    def __init__(
        self, 
        model: BaseRecommender,
        loader: BaseItemLoader,
    ):
        self.model = model
        self.loader = loader

    @abstractmethod
    def predict(
        self,
        outfits: List[Outfit],
    ) -> List[float]:
        
        raise NotImplementedError(
            'The cp method must be implemented by subclasses.'
        )
        
        
class BaseCIRPipeline(ABC):
    
    def __init__(
        self,
        model: BaseRecommender,
        loader: BaseItemLoader,
        indexer: BaseIndexer
    ):
        self.model = model
        self.loader = loader
        self.indexer = indexer

    @abstractmethod
    def search(
        self,
        queries: List[Query],
        k: int,
    ) -> List[List[Item]]:
        
        raise NotImplementedError(
            'The search method must be implemented by subclasses.'
        )