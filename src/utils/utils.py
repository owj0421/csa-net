import os
import random
import numpy as np
import torch
from typing import Iterable, Any, Optional
from itertools import islice
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def batch_iterable(
    iterable: Iterable[Any],
    batch_size: int,
    desc: Optional[str] = None,
):
    iterator = iter(iterable)
    total = len(iterable) if hasattr(iterable, '__len__') else None
    
    pbar = tqdm(
        total=(total + batch_size - 1) // batch_size if total else None,
        desc=desc,
    )
    
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
        if pbar.total:  # Update progress only if total is known
            pbar.update(1)
            
            
def get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def aggregate_embeddings(
    image_embeddings: Optional[Tensor] = None, 
    text_embeddings: Optional[Tensor] = None, 
    aggregation_method: str = 'concat'
) -> Tensor:
    embeds = []
    if image_embeddings is not None:
        embeds.append(image_embeddings)
    if text_embeddings is not None:
        embeds.append(text_embeddings)

    if not embeds:
        raise ValueError('At least one of image_embeds or text_embeds must be provided.')

    if aggregation_method == 'concat':
        return torch.cat(embeds, dim=-1)
    elif aggregation_method == 'mean':
        return torch.mean(torch.stack(embeds), dim=-2)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}. Use 'concat' or 'mean'.")


def mean_pooling(
    model_output: Tensor, 
    attention_mask: Tensor
) -> Tensor:
    token_embeddings = model_output[0]  # First element of model_output contains the hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    mask_sum = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return summed_embeddings / mask_sum