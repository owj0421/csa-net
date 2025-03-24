from typing import List
import torch
from torch.distributed import get_rank, get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP

from .csa_net import (
    CSANetConfig, 
    CSANetEncoder, 
    CSANetSubspaceAttention
)


def ddp_to_std(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict


def load_model(checkpoint=None, **cfg_kwargs):
    is_distributed = torch.distributed.is_initialized()
    
    if is_distributed:
        rank = get_rank()
    else:
        rank = 0
        
    map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        
    if checkpoint:
        loaded = torch.load(checkpoint, map_location=map_location)
        
        cfg = CSANetConfig(**loaded.get('CSANetConfig', {}))
        enc, sp_attn = CSANetEncoder(cfg), CSANetSubspaceAttention(cfg)
        
        enc.load_state_dict(ddp_to_std(loaded.get('CSANetEncoder', {})))
        sp_attn.load_state_dict(ddp_to_std(loaded.get('CSANetSubspaceAttention', {})))
    else:
        cfg = CSANetConfig(**cfg_kwargs)
        enc, sp_attn = CSANetEncoder(cfg), CSANetSubspaceAttention(cfg)
    
    enc.to(map_location)
    sp_attn.to(map_location)
    
    if is_distributed:
        enc = DDP(enc, device_ids=[rank], static_graph=True)
        sp_attn = DDP(sp_attn, device_ids=[rank], static_graph=True)
    
    return cfg, enc, sp_attn
    