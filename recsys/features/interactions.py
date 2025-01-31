def process_interactions(bets_df, valid_user_ids, valid_event_ids):
    
    # Filter for valid interactions
    valid_bets = bets_df[
        bets_df['player_id'].isin(valid_user_ids) & 
        bets_df['event_id'].isin(valid_event_ids)
    ]
    return set(zip(valid_bets['player_id'], valid_bets['event_id']))