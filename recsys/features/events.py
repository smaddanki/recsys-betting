import pandas as pd
from sklearn.preprocessing import LabelEncoder

def process_event_features(events_df):
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
    event_feature_matrix = feature_matrix.to_numpy()
    event_id_to_idx = {id_: idx for idx, id_ in enumerate(feature_matrix.index)}
    
    return feature_matrix, event_feature_matrix


def get_valid_events(events_df):
    
    return set(events_df['event_id'])