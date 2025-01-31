from scipy import sparse
import numpy as np


def create_labels(interactions, valid_user_ids, valid_event_ids, negative_samples = 4):
    # Convert interactions to numpy for faster operations
    interactions_array = np.array(list(interactions))
    positive_array = interactions_array.copy()
    
    # Pre-allocate arrays for negative samples
    n_users = len(valid_user_ids)
    n_neg_samples = n_users * negative_samples
    negative_array = np.zeros((n_neg_samples, 2), dtype=interactions_array.dtype)
    
    # Create interaction matrix for fast lookup
    user_event_matrix = sparse.coo_matrix(
        (
            np.ones(len(interactions_array)),
            (interactions_array[:, 0], interactions_array[:, 1])
        ),
        shape=(max(valid_user_ids) + 1, max(valid_event_ids) + 1)
    ).tocsr()
    
    # Vectorized negative sampling
    idx = 0
    events_array = np.array(list(valid_event_ids))
    
    for user_id in valid_user_ids:
        # Get user interactions
        user_interactions = user_event_matrix[user_id].indices
        
        # Create mask for available events
        mask = np.ones(len(events_array), dtype=bool)
        mask[np.searchsorted(events_array, user_interactions)] = False
        available_events = events_array[mask]
        
        if len(available_events) > 0:
            # Sample negative events
            n_samples = min(negative_samples, len(available_events))
            sampled_events = np.random.choice(
                available_events, 
                size=n_samples, 
                replace=False
            )
            
            # Store negative samples
            end_idx = idx + n_samples
            negative_array[idx:end_idx, 0] = user_id
            negative_array[idx:end_idx, 1] = sampled_events
            idx += n_samples
    
    # Trim unused allocation
    negative_array = negative_array[:idx]
    
    # Combine positive and negative samples
    all_samples = np.vstack([positive_array, negative_array])
    labels = np.hstack([
        np.ones(len(positive_array)), 
        np.zeros(len(negative_array))
    ])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(all_samples))
    all_samples = all_samples[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return list(zip(map(tuple, all_samples), labels))