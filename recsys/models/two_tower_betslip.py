
import torch.nn as nn
import torch

class BetslipRecommenderModel(nn.Module):
    def __init__(self, user_dim, event_dim, market_dim, embedding_dim=64):
        super().__init__()
        
        # User Tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        # Event Tower
        self.event_tower = nn.Sequential(
            nn.Linear(event_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        # Market Tower
        self.market_tower = nn.Sequential(
            nn.Linear(market_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        
        # Betslip Composer
        self.composer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=256
            ),
            num_layers=2
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, user_features, event_features_list, market_features_list):
        # Get user embedding
        user_emb = self.user_tower(user_features)
        
        # Process each selection in betslip
        selection_embeddings = []
        for event_feat, market_feat in zip(event_features_list, market_features_list):
            event_emb = self.event_tower(event_feat)
            market_emb = self.market_tower(market_feat)
            # Combine event and market embeddings
            selection_emb = event_emb + market_emb
            selection_embeddings.append(selection_emb)
        
        # Stack selection embeddings
        selections = torch.stack(selection_embeddings)
        
        # Combine selections using transformer
        betslip_emb = self.composer(selections)
        betslip_emb = torch.mean(betslip_emb, dim=0)
        
        # Final prediction
        score = self.predictor(betslip_emb + user_emb)
        return score