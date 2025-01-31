import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    user_features = torch.stack([item['user_features'] for item in batch])
    event_features = torch.stack([item['event_features'] for item in batch])
    interactions = torch.stack([item['interaction'] for item in batch])
    
    return {
        'user_features': user_features,
        'event_features': event_features,
        'interaction': interactions
    }

def get_train_val_split(features: Dataset, batch_size: int=32, train_split= 0.8):
    """
    Prepare data for the model
    """

    # Split indices for train/val
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    split_idx = int(len(features) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create samplers
    from torch.utils.data import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        features, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        features, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader