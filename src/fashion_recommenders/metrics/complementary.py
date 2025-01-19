import numpy as np
import typing 
from sklearn.metrics import roc_auc_score
from typing import List

def compute_accuracy(
    predictions: np.array,
    labels: np.array
):
    return (predictions == labels).mean()

    
class ComplementaryScore(typing.TypedDict):
    acc = typing.Union[float, None]
    

class ComplementaryMetricCalculator:
    def __init__(self):
        self.__predictions = np.array([], dtype=int)
        self.__labels = np.array([], dtype=int)

    def add(
        self, 
        query_embeddings: List[np.array],  # batch_size list of (n_items, embedding_dim)
        candidate_embeddings: List[np.array],  # batch_size list of (num_candidates, embedding_dim)
        labels: np.array  # (batch_size,)
    ):
        batch_sz = len(query_embeddings)
        
        # Compute pairwise distances in a vectorized manner
        dists = np.array([
            np.sum(
                np.linalg.norm(qs[:, None, :] - cs[None, :, :], axis=2), axis=0
            )
            for qs, cs in zip(query_embeddings, candidate_embeddings)
        ])  # Shape: (batch_sz, num_candidates)
        
        predictions = dists.argmin(axis=1)  # Get the index of the closest candidate
        self.__predictions = np.append(self.__predictions, predictions)
        self.__labels = np.append(self.__labels, labels)
        
        return ComplementaryScore(
            acc=compute_accuracy(predictions, labels),
        )
        
    def reset(self):
        self.__predictions = np.array([])
        self.__labels = np.array([])
    
    
    def calculate(self):
        return ComplementaryScore(
            acc=compute_accuracy(self.__predictions, self.__labels),
        )