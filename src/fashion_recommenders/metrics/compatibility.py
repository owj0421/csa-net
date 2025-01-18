import numpy as np
import typing 
from sklearn.metrics import roc_auc_score



def compute_scores(
    predictions: np.array,
    labels: np.array
):
    predictions = (predictions > 0.5).astype(int)
    
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    
    accuracy = (predictions == labels).mean()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (labels == 1).sum() if (labels == 1).sum() > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return {
        'acc': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1
    }


def compute_auc(
    predictions: np.array,
    labels: np.array
):
    if len(np.unique(labels)) == 1:
        return 0.0
    
    return roc_auc_score(
        y_true=labels,
        y_score=predictions
    )
    
    
class CompatibilityScore(typing.TypedDict):
    auc = typing.Union[float, None]
    acc = typing.Union[float, None]
    precision = typing.Union[float, None]
    recall = typing.Union[float, None]
    f1 = typing.Union[float, None]
    

class CompatibilityMetricCalculator():
    __predictions = np.array([])
    __labels = np.array([])

    def add(
        self, 
        predictions: np.array,
        labels: np.array
    ):
        self.__predictions = np.concatenate([self.__predictions, predictions])
        self.__labels = np.concatenate([self.__labels, labels])
        
        return CompatibilityScore(
            auc=compute_auc(predictions, labels),
            **compute_scores(predictions, labels)
        )
        
    def reset(self):
        self.__predictions = np.array([])
        self.__labels = np.array([])
    
    
    def calculate(self, reset=True):
        score = CompatibilityScore(
            auc=compute_auc(self.__predictions, self.__labels),
            **compute_scores(self.__predictions, self.__labels)
        )
        self.reset()
        
        return score