from torch.utils.data import Dataset
import torch
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class BettingDataset(Dataset):
    def __init__(self, user_df, event_df, bets_df, negative_samples=4):
        self.negative_samples = negative_samples
        
        # Store user and event ids for sampling
        self.valid_user_ids = set(user_df['player_id'])
        self.valid_event_ids = set(event_df['event_id'])
        
        # Process features
        self.user_features = self.process_user_features(user_df, bets_df)
        self.event_features = self.process_event_features(event_df)
        
        # Create positive interactions
        self.interactions = self.process_interactions(bets_df)
        

        
        # Create training samples
        self.training_samples = self.create_training_samples()
        
        print(f"Dataset created with {len(self.training_samples)} samples")
        print(f"Valid users: {len(self.valid_user_ids)}")
        print(f"Valid events: {len(self.valid_event_ids)}")
        
    def process_user_features(self, users_df, bets_df):
        # Calculate user statistics
        user_stats = bets_df.groupby('player_id').agg({
            'amount': ['mean', 'count'],
            'bet_odds': 'mean',
            'status': lambda x: (x == 'won').mean()
        }).reset_index()
        
        user_stats.columns = ['player_id', 'avg_stake', 'bet_count', 'avg_odds', 'win_rate']
        
        # Merge with user data
        users_df = users_df.merge(user_stats, on='player_id', how='left')
        
        # Fill NaN values
        numerical_features = ['avg_stake', 'bet_count', 'avg_odds', 'win_rate']
        users_df[numerical_features] = users_df[numerical_features].fillna(0)
        
        # Add temporal features
        users_df['days_since_reg'] = (pd.to_datetime('now') - pd.to_datetime(users_df['player_reg_date'])).dt.days
        
        # Process categorical features
        categorical_features = ['language', 'brand_id']
        for feature in categorical_features:
            le = LabelEncoder()
            users_df[feature + '_encoded'] = le.fit_transform(users_df[feature].astype(str))
        
        # Select features
        features = numerical_features + ['days_since_reg'] + [f + '_encoded' for f in categorical_features]
        
        # Create feature matrix indexed by player_id
        feature_matrix = users_df.set_index('player_id')[features]
        
        # Convert to numpy and ensure 2D
        self.user_feature_matrix = feature_matrix.to_numpy().reshape(len(feature_matrix), -1)
        self.user_id_to_idx = {id_: idx for idx, id_ in enumerate(feature_matrix.index)}
        
        return self.user_feature_matrix
    
    def process_event_features(self, events_df):
        # Process temporal features
        events_df['time_to_event'] = (pd.to_datetime(events_df['start_time']) - pd.to_datetime('now')).dt.total_seconds() / 3600
        
        # Process categorical features
        categorical_features = ['sport_id', 'league_id']
        for feature in categorical_features:
            le = LabelEncoder()
            events_df[feature + '_encoded'] = le.fit_transform(events_df[feature].astype(str))
        
        # Select features
        features = ['time_to_event'] + [f + '_encoded' for f in categorical_features]
        
        # Create feature matrix indexed by event_id
        feature_matrix = events_df.set_index('event_id')[features]
        
        # Convert to numpy for faster access
        self.event_feature_matrix = feature_matrix.to_numpy()
        self.event_id_to_idx = {id_: idx for idx, id_ in enumerate(feature_matrix.index)}
        
        return feature_matrix

    def process_interactions(self, bets_df):
        # Filter for valid interactions
        valid_bets = bets_df[
            bets_df['player_id'].isin(self.valid_user_ids) & 
            bets_df['event_id'].isin(self.valid_event_ids)
        ]
        return set(zip(valid_bets['player_id'], valid_bets['event_id']))
    
    def create_training_samples(self):
        # Convert interactions to numpy for faster operations
        interactions_array = np.array(list(self.interactions))
        positive_samples = interactions_array.copy()
        
        # Pre-allocate arrays for negative samples
        n_users = len(self.valid_user_ids)
        n_neg_samples = n_users * self.negative_samples
        negative_samples = np.zeros((n_neg_samples, 2), dtype=interactions_array.dtype)
        
        # Create interaction matrix for fast lookup
        user_event_matrix = sparse.coo_matrix(
            (
                np.ones(len(interactions_array)),
                (interactions_array[:, 0], interactions_array[:, 1])
            ),
            shape=(max(self.valid_user_ids) + 1, max(self.valid_event_ids) + 1)
        ).tocsr()
        
        # Vectorized negative sampling
        idx = 0
        events_array = np.array(list(self.valid_event_ids))
        
        for user_id in self.valid_user_ids:
            # Get user interactions
            user_interactions = user_event_matrix[user_id].indices
            
            # Create mask for available events
            mask = np.ones(len(events_array), dtype=bool)
            mask[np.searchsorted(events_array, user_interactions)] = False
            available_events = events_array[mask]
            
            if len(available_events) > 0:
                # Sample negative events
                n_samples = min(self.negative_samples, len(available_events))
                sampled_events = np.random.choice(
                    available_events, 
                    size=n_samples, 
                    replace=False
                )
                
                # Store negative samples
                end_idx = idx + n_samples
                negative_samples[idx:end_idx, 0] = user_id
                negative_samples[idx:end_idx, 1] = sampled_events
                idx += n_samples
        
        # Trim unused allocation
        negative_samples = negative_samples[:idx]
        
        # Combine positive and negative samples
        all_samples = np.vstack([positive_samples, negative_samples])
        labels = np.hstack([
            np.ones(len(positive_samples)), 
            np.zeros(len(negative_samples))
        ])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(all_samples))
        all_samples = all_samples[shuffle_idx]
        labels = labels[shuffle_idx]
        
        return list(zip(map(tuple, all_samples), labels))
    
    def __len__(self):
        return len(self.training_samples)
    
    def __getitem__(self, idx):
        (user_id, event_id), label = self.training_samples[idx]
        
        try:
            # Get features using numpy arrays and ensure proper shapes
            user_features = self.user_feature_matrix[self.user_id_to_idx[user_id]].flatten()
            event_features = self.event_feature_matrix[self.event_id_to_idx[event_id]].flatten()
            
            return {
                'user_features': torch.FloatTensor(user_features),
                'event_features': torch.FloatTensor(event_features),
                'interaction': torch.FloatTensor([label])
            }
        except Exception as e:
            print(f"Error at idx {idx}: user_id {user_id}, event_id {event_id}")
            print(f"User features shape: {user_features.shape if 'user_features' in locals() else 'Not created'}")
            print(f"Event features shape: {event_features.shape if 'event_features' in locals() else 'Not created'}")
            raise e