import json
import logging
import os
import pathlib
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..evaluation.metrics import compute_cir_scores, compute_cp_scores
from ..models.load import load_model
from ..models.csa_net import CSANetEncoder, CSANetSubspaceAttention, CSANetConfig
from ..models.loss import InBatchTripletMarginLoss
from ..utils.distributed_utils import cleanup, gather_results, setup
import random
from PIL import Image


SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Constants
MAX_OUTFIT_LENGTH = 10
PAD_IMAGE = Image.new("RGB", (224, 224))
PAD_CATEGORY = 'unknown'
# ----------------------------
import torch
torch.autograd.set_detect_anomaly(True)

class InBatchOutfitRankingLoss(nn.Module):
    
    def __init__(self, margin: float = 2.0, reduction: str = 'mean', return_score: bool = True):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.return_score = return_score
        
    def forward(self, w_idx_dict, w, batched_anc_cat, batched_anc_emb, batched_pos_cat, batched_pos_emb, mask):
        """_summary_

        Args:
            w_dict (_type_): Dictionary of Subspace Attention Weights, Dictionary with {str: tensor of dim (n_sp, d_emb)}
            batched_anc_cat (_type_): Categories of Items in Query Outfit, List of string (bsz, max_length)
            batched_anc_emb (_type_): Embedding of Items in Query Outfit with Subspace, dim: (bsz, max_length, n_sp, d_emb)
            batched_pos_cat (_type_): Categories of Items in Answer Outfit, List of string (bsz, 1)
            batched_pos_emb (_type_): Embedding of Items in Answer Outfit with Subspace, dim: (bsz, n_sp, d_emb)
        """
        bsz = batched_anc_emb.shape[0]
        max_length = batched_anc_emb.shape[1]
        n_sp = batched_anc_emb.shape[2]
        d_emb = batched_anc_emb.shape[3]
        
        batched_anc_emb = batched_anc_emb\
            .view(bsz, 1, max_length, n_sp,  d_emb)
        
        batched_cs_emb = batched_pos_emb\
            .view(bsz, 1, n_sp, d_emb).expand(bsz, bsz, n_sp, d_emb)\
            .unsqueeze(2).expand(bsz, bsz, max_length, n_sp, d_emb)
            
        mask = torch.eye(bsz, dtype=torch.bool)
        
        batched_pos_emb = batched_cs_emb[mask]\
            .view(bsz, 1, max_length, n_sp, d_emb)
            
        batched_neg_cand_emb = batched_cs_emb[~mask]\
            .view(bsz, bsz-1, max_length, n_sp, d_emb)
            
        ans_pos_cat = [(batched_anc_cat[b_i][anc_i], batched_pos_cat[b_i][0]) for anc_i in range(max_length) for b_i in range(bsz)]
        w = torch.stack([w[w_idx_dict[i]] for i in ans_pos_cat], dim=0)\
            .view(bsz, 1, max_length, n_sp, 1)

        batched_anc_emb = torch.sum(batched_anc_emb * w, dim=-2)
        batched_pos_emb = torch.sum(batched_pos_emb * w, dim=-2)
        batched_neg_cand_emb = torch.sum(batched_neg_cand_emb * w.expand(bsz, bsz-1, max_length, n_sp, 1), dim=-2)
        
        ans_pos_dist = torch.sum(torch.norm(batched_anc_emb - batched_pos_emb, p=2, dim=-1), dim=-1)  # (bsz, 1, max_length) -> (bsz, 1)
        ans_neg_cand_dist = torch.sum(torch.norm(batched_pos_emb - batched_neg_cand_emb, p=2, dim=-1), dim=-1)  # (bsz, bsz-1, max_length) -> (bsz, bsz-1)
        ans_neg_dist = ans_neg_cand_dist.gather(1, torch.argmin(ans_neg_cand_dist, dim=-1, keepdim=True))  # (bsz, bsz-1) -> (bsz, 1)
        
        loss = torch.clamp(ans_pos_dist - ans_neg_dist + self.margin, min=0.0)
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = loss
            
        if not self.return_score:
            return loss
        
        # bsz별로 Pos_dist가 ans_neg_cand_dist의 모든 값보다 작은지 확인
        candidates = torch.cat([ans_pos_dist.detach().cpu(), ans_neg_cand_dist.detach().cpu()], dim=-1)
        preds = torch.argmin(candidates, dim=-1)
        labels = torch.zeros(bsz, dtype=torch.long)
        
        return loss, preds, labels


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=4)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--n_epochs', type=int,
                        default=200)
    parser.add_argument('--lr', type=float,
                        default=2e-5)
    parser.add_argument('--accumulation_steps', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--world_size', type=int, 
                        default=1)
    parser.add_argument('--project_name', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()

# Utils for Save and Load
def save_checkpoint(cfg, enc, sp_attn, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
    
    torch.save({
        'CSANetConfig': cfg.__dict__,
        'CSANetEncoder': enc.state_dict(),
        'CSANetSubspaceAttention': sp_attn.state_dict(),
    }, checkpoint_path)
    
    return checkpoint_path
    
    
def save_results(train_log, valid_log, checkpoint_dir, epoch):
    results_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
    
    with open(results_path, 'w') as f:
        results = {**train_log, **valid_log}
        json.dump(results, f, indent=4)
        
    return results_path


def load_checkpoint(checkpoint_path, enc, sp_attn, rank):
    map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    enc = enc.load_state_dict(state_dict['CSANetEncoder'])
    sp_atten = sp_attn.load_state_dict(state_dict['CSANetSubspaceAttention'])
    
    return enc, sp_attn
# ----------------------------

# Utils for Model
def preprocess(outfits):
    
    def get_max_len_of_sequences(sequences):
        return min(max(len(seq) for seq in sequences), MAX_OUTFIT_LENGTH)

    def pad_sequences(sequences, pad_value, max_length):
        return [seq[:max_length] + [pad_value] * (max_length - len(seq)) for seq in sequences]
    
    max_length = get_max_len_of_sequences(outfits)
    
    images = [
        [item.image for item in seq] + [PAD_IMAGE] * (max_length - len(seq)) 
        for seq in outfits
    ]
    
    categories = [
        [item.category for item in seq] + [PAD_CATEGORY] * (max_length - len(seq)) 
        for seq in outfits
    ]
    
    mask = [
        [0] * len(seq) + [1] * (max_length - len(seq)) 
        for seq in outfits
    ]
    
    return images, categories, torch.BoolTensor(mask), max_length



# ----------------------------

# Utils
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ----------------------------

# Dataset
def setup_dataloaders(rank, world_size, args):    
    metadata = polyvore.load_metadata(args.polyvore_dir)
    
    train = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='train', metadata=metadata, load_image=True
    )
    valid = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='valid', metadata=metadata, load_image=True
    )
    
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn
        )
        
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn, sampler=train_sampler
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn, sampler=valid_sampler
        )

    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, wandb_run,
    cfg: CSANetConfig, enc:CSANetEncoder, sp_attn:CSANetSubspaceAttention, optimizer, scheduler, loss_fn, dataloader
):
    enc.train(); sp_attn.train()
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}')
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        
        cat = cfg.category
        bsz = args.batch_sz_per_gpu
        d_emb = cfg.d_embed
        n_sp = cfg.n_subspace
        
        # qs: query item"s"
        # a: answer item
        # cs: candidate item"s"
        batched_qs = [q.outfit for q in data['query']]
        batched_a = data['answer']
        
        comb_of_cat = [(cat1, cat2) for cat1 in cat for cat2 in cat]
        w_idx_dict, w = sp_attn(comb_of_cat) # (i.e. ('hat', 'sunglass')): (Array of dim [n_subspace])
        
        batched_anc_img, batched_anc_cat, mask, max_length = preprocess(batched_qs)
        batched_pos_img, batched_pos_cat, _, _ = preprocess([[a] for a in batched_a])
        
        batched_anc_emb = enc(sum(batched_anc_img, [])).view(bsz, max_length, n_sp, d_emb)
        batched_pos_emb = enc(sum(batched_pos_img, [])).view(bsz, n_sp, d_emb)
        
        loss, preds, labels = loss_fn(w_idx_dict, w, batched_anc_cat, batched_anc_emb, batched_pos_cat, batched_pos_emb, mask)
        
        loss = loss / args.accumulation_steps
        
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        score = compute_cir_scores(preds, labels)
        
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'train_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {'loss': gathered_loss.item(), **compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    output = {f'train_{key}': value for key, value in output.items()}
    print(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return output


# @torch.no_grad()
# def valid_step(
#     rank, world_size, 
#     args, epoch, wandb_run,
#     enc, sp_attn, loss_fn, dataloader
# ):
#     model.eval()
#     pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}')
    
#     all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
#     for i, data in enumerate(pbar):
#         if args.demo and i > 2:
#             break
#         batched_q_emb = model(data['query'], use_precomputed_embedding=True).unsqueeze(1) # (batch_sz, 1, embedding_dim)
#         batched_c_embs = model(sum(data['candidates'], []), use_precomputed_embedding=True) # (batch_sz * 4, embedding_dim)
#         batched_c_embs = batched_c_embs.view(-1, 4, batched_c_embs.shape[1]) # (batch_sz, 4, embedding_dim)
        
#         dists = torch.norm(batched_q_emb - batched_c_embs, dim=-1) # (batch_sz, 4)
#         preds = torch.argmin(dists, dim=-1) # (batch_sz,)
#         labels = torch.tensor(data['label'], device=rank)

#         # Accumulate Results
#         all_preds.append(preds.detach())
#         all_labels.append(labels.detach())

#         # Logging
#         score = compute_cir_scores(all_preds[-1], all_labels[-1])
#         logs = {
#             'steps': len(pbar) * epoch + i,
#             **score
#         }
#         pbar.set_postfix(**logs)
#         if args.wandb_key and rank == 0:
#             logs = {f'valid_{k}': v for k, v in logs.items()}
#             wandb_run.log(logs)
    
#     all_preds = torch.cat(all_preds).to(rank)
#     all_labels = torch.cat(all_labels).to(rank)

#     _, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
#     output = {**compute_cir_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
#     output = {f'valid_{key}': value for key, value in output.items()}
#     print(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

#     return output

    
def train(
    rank: int, world_size: int, args: Any,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
):      
    # Setup
    setup(rank, world_size)
    
    # Logging Setup
    project_name = f'complementary_' + (
        args.project_name if args.project_name 
        else (wandb_run.name if wandb_run else 'test')
    )
    print(f'Logger Setup Completed')
    
    checkpoint_dir = CHECKPOINT_DIR / project_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataloaders
    train_dataloader, valid_dataloader = setup_dataloaders(rank, world_size, args)
    print(f'Dataloaders Setup Completed')
    
    # Model setting
    cfg, enc, sp_attn = load_model(checkpoint=args.checkpoint)
    print(f'Model Loaded and Wrapped with DDP')
    
    # Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.AdamW(
        list(enc.parameters()) + list(sp_attn.parameters()), 
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=int(len(train_dataloader) / args.accumulation_steps),
        pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
    )
    loss_fn = InBatchOutfitRankingLoss(margin=2.0, reduction='mean')
    print(f'Optimizer and Scheduler Setup Completed')

    # Training Loop
    for epoch in range(args.n_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)
            
        train_logs = train_step(
            rank, world_size, 
            args, epoch, wandb_run,
            cfg, enc, sp_attn, optimizer, scheduler, loss_fn, train_dataloader
        )

        # valid_logs = valid_step(
        #     rank, world_size, 
        #     args, epoch, wandb_run,
        #     cfg, enc, sp_attn, loss_fn, valid_dataloader
        # )
        valid_logs = None
            
        if rank == 0:
            checkpoint_path = save_checkpoint(args, enc, sp_attn, checkpoint_dir, epoch)
            _ = save_results(train_logs, valid_logs, checkpoint_dir, epoch)
            print(f'Checkpoint saved at {checkpoint_path}')
            
        dist.barrier()
        enc, sp_attn = load_checkpoint(checkpoint_path, enc, sp_attn, rank)

    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(project='outfit-transformer-cir', config=args.__dict__)
    else:
        wandb_run = None
    
    mp.spawn(
        train, args=(args.world_size, args, wandb_run), 
        nprocs=args.world_size, join=True
    )