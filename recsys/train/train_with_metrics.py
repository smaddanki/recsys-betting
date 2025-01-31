import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def calculate_metrics(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get batch data
            user_features = batch['user_features'].to(device)
            event_features = batch['event_features'].to(device)
            labels = batch['interaction'].to(device)
            
            # Forward pass
            outputs = model(user_features, event_features)
            
            # Calculate loss (Binary Cross Entropy)
            loss = criterion(outputs, labels.squeeze())
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()  # Threshold at 0.5
            correct += (predictions == labels.squeeze()).sum().item()
            total += labels.size(0)
            
            # Store predictions and labels for additional metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    # Calculate additional metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_predictions)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    metrics_history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'train_auc': [], 'val_auc': []
    }
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_metrics = calculate_metrics(model, train_loader, criterion, device)
        
        # Validation Phase
        model.eval()
        val_metrics = calculate_metrics(model, val_loader, criterion, device)
        
        # Store metrics
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['train_acc'].append(train_metrics['accuracy'])
        metrics_history['train_auc'].append(train_metrics['auc'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['val_acc'].append(val_metrics['accuracy'])
        metrics_history['val_auc'].append(val_metrics['auc'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"Train Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
        print(f"Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print("-" * 50)
    
    return metrics_history

# Visualize metrics
def plot_metrics(metrics_history):
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(metrics_history['train_acc'], label='Train Accuracy')
    plt.plot(metrics_history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(metrics_history['train_auc'], label='Train AUC')
    plt.plot(metrics_history['val_auc'], label='Val AUC')
    plt.title('AUC over epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()