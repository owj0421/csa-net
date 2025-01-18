
from abc import ABC, abstractmethod
from typing import List
from . import datatypes
        
        
class BasePipeline(ABC):
    
    @abstractmethod
    def compatibility_predict(
        self,
        queries: List[datatypes.FashionCompatibilityQuery],
    ) -> List[float]:
        
        raise NotImplementedError(
            'The cp method must be implemented by subclasses.'
        )

    @abstractmethod
    def complementary_search(
        self,
        queries: List[datatypes.FashionComplementaryQuery],
        k: int,
    ) -> List[List[datatypes.FashionItem]]:
        
        raise NotImplementedError(
            'The cir method must be implemented by subclasses.'
        )