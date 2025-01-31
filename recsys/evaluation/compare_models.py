def compare_models(two_tower_model, mind_model, test_data):
    metrics = {
        'accuracy': [],
        'ndcg': [],
        'map': [],
        'recall': [],
        'diversity': [],
        'coverage': [],
        'response_time': []
    }
    
    # Performance Metrics
    for model in [two_tower_model, mind_model]:
        predictions = model.predict(test_data)
        metrics['accuracy'].append(calculate_accuracy(predictions))
        metrics['ndcg'].append(calculate_ndcg(predictions))
        metrics['map'].append(calculate_map(predictions))
        metrics['recall'].append(calculate_recall(predictions))
        
        # Diversity and Coverage
        metrics['diversity'].append(calculate_diversity(predictions))
        metrics['coverage'].append(calculate_coverage(predictions))
        
        # Inference Time
        metrics['response_time'].append(measure_inference_time(model))
    
    return metrics