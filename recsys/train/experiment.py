import torch
from ..config import device



# Train model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.train()
    total_steps = len(train_loader)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for i, batch in enumerate(train_loader):
            user_features = batch['user_features'].to(device)
            event_features = batch['event_features'].to(device)
            labels = batch['interaction'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(user_features, event_features)
            loss = criterion(outputs, labels.squeeze())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0).float()
            correct += (predictions == labels.squeeze()).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Accuracy: {100 * correct/total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0
                
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_features = batch['user_features'].to(device)
                event_features = batch['event_features'].to(device)
                labels = batch['interaction'].to(device)
                
                outputs = model(user_features, event_features)
                loss = criterion(outputs, labels.squeeze())
                
                val_loss += loss.item()
                predictions = (outputs > 0).float()
                val_correct += (predictions == labels.squeeze()).sum().item()
                val_total += labels.size(0)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {100 * val_correct/val_total:.2f}%')
        
        model.train()