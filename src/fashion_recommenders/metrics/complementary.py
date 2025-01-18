import numpy as np
import typing 
from sklearn.metrics import roc_auc_score


def compute_accuracy(
    predictions: np.array,
    labels: np.array
):
    return (predictions == labels).mean()

    
class ComplementaryScore(typing.TypedDict):
    acc = typing.Union[float, None]
    

class ComplementaryMetricCalculator():
    __predictions = np.array([])
    __labels = np.array([])

    def add(
        self, 
        query_embeddings: np.array, # (batch_size, embedding_dim)
        candidate_embeddings: np.array, # (batch_size, num_candidates, embedding_dim)
        labels: np.array # (batch_size)
    ):
        if query_embeddings.ndim == 1:
            query_embeddings = np.expand_dims(query_embeddings, axis=0)
        if candidate_embeddings.ndim == 2:
            candidate_embeddings = np.expand_dims(candidate_embeddings, axis=0)
        
        batch_sz, n_candidates, d_embedding = candidate_embeddings.shape
        
        query_embeddings_repeated = np.repeat(query_embeddings[:, np.newaxis, :], n_candidates, axis=1)
        query_embeddings_reshaped = query_embeddings_repeated.reshape(-1, d_embedding)
        candidate_embeddings_reshaped = candidate_embeddings.reshape(-1, d_embedding)
        query_cand_dist = np.linalg.norm(query_embeddings_reshaped - candidate_embeddings_reshaped, axis=1)
        dist = query_cand_dist.reshape(batch_sz, n_candidates)
        
        predictions = dist.argmin(axis=1)
        
        self.__predictions = np.concatenate([self.__predictions, predictions])
        self.__labels = np.concatenate([self.__labels, labels])
        
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