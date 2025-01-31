
import torch.nn as nn
import torch

class TwoTowerModel(nn.Module):
    def __init__(self, user_features_dim, event_features_dim, embedding_dim=64, dropout_rate=0.2):
        super(TwoTowerModel, self).__init__()
        
        # User Tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_features_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Event Tower
        self.event_tower = nn.Sequential(
            nn.Linear(event_features_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, user_features, event_features):
        # Get embeddings from both towers
        user_embedding = self.user_tower(user_features)
        event_embedding = self.event_tower(event_features)
        
        # Compute similarity
        similarity = torch.sum(user_embedding * event_embedding, dim=1)
        return similarity

def initialize_model(dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Get feature dimensions from the dataset
    user_features_dim = dataset.user_feature_matrix.shape[1]
    event_features_dim = dataset.event_feature_matrix.shape[1]
    
    print(f"User features dimension: {user_features_dim}")
    print(f"Event features dimension: {event_features_dim}")
    
    # Initialize model
    model = TwoTowerModel(
        user_features_dim=user_features_dim,
        event_features_dim=event_features_dim,
        embedding_dim=64,  # You can adjust this
        dropout_rate=0.2   # You can adjust this
    )
    
    # Move model to device
    model = model.to(device)
    
    return model

# Training configuration
def setup_training(model, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return criterion, optimizer