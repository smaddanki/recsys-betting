import torch
import pandas as pd
from ..config import device

def generate_recommendations(model, dataset, user_id, top_k=10, device=device):
    
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        try:
            # Get user features
            user_idx = dataset.user_id_to_idx[user_id]
            user_features = dataset.user_feature_matrix[user_idx]
            user_features = torch.FloatTensor(user_features).unsqueeze(0).to(device)
            
            # Get all event features
            all_event_features = torch.FloatTensor(dataset.event_feature_matrix).to(device)
            
            # Expand user features to match event features dimension
            user_features_expanded = user_features.expand(len(dataset.event_features), -1)
            
            # Get scores for all events
            scores = model(user_features_expanded, all_event_features)
            
            # Get top-k recommendations
            top_scores, top_indices = torch.topk(scores, k=top_k)
            
            # Convert indices back to event IDs
            event_ids = list(dataset.event_id_to_idx.keys())
            recommended_events = [event_ids[idx] for idx in top_indices.cpu().numpy()]
            recommendation_scores = top_scores.cpu().numpy()
            
            # Create recommendations dataframe
            recommendations = pd.DataFrame({
                'event_id': recommended_events,
                'score': recommendation_scores
            })
            
            return recommendations
            
        except KeyError:
            print(f"User ID {user_id} not found in the dataset")
            return None

# Function to get event details
def get_event_details(recommendations, events_df):
    """
    Add event details to recommendations
    """
    return recommendations.merge(
        events_df[['event_id', 'sport_id', 'league_id', 'home_team', 'away_team', 'start_time']], 
        on='event_id',
        how='left'
    )

# Example usage:
def get_user_recommendations(model, dataset, user_id, events_df, top_k=10):
    """
    Get recommendations with full event details
    """
    # Get raw recommendations
    recommendations = generate_recommendations(model, dataset, user_id, top_k=top_k)
    
    if recommendations is not None:
        # Add event details
        detailed_recommendations = get_event_details(recommendations, events_df)
        
        # Sort by score
        detailed_recommendations = detailed_recommendations.sort_values('score', ascending=False)
        
        # Format score
        detailed_recommendations['score'] = detailed_recommendations['score'].round(4)
        
        return detailed_recommendations
    return None