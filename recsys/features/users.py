import pandas as pd
from sklearn.preprocessing import LabelEncoder


def process_user_features( users_df, bets_df):
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
    user_feature_matrix = feature_matrix.to_numpy().reshape(len(feature_matrix), -1)
    user_id_to_idx = {id_: idx for idx, id_ in enumerate(feature_matrix.index)}
    
    return feature_matrix, user_feature_matrix

def get_valid_users(users_df):
    
    return set(users_df['player_id'])
    