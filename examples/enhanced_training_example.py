"""
Example of how to integrate InterpretabilityMonitor into training loop.

This shows how to modify the training process to monitor interpretability
metrics in real-time rather than post-hoc analysis.
"""

from interpretability_monitor import InterpretabilityMonitor

def enhanced_training_loop(model, train_loader, optimizer, num_epochs):
    """
    Enhanced training loop with interpretability monitoring.
    """
    # Initialize interpretability monitor
    interp_monitor = InterpretabilityMonitor(model, log_frequency=50)
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (questions, responses) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass - MODIFIED to capture intermediate outputs
            predictions, context_seq, value_seq = model.forward_with_states(questions, responses)
            
            # Get projected outputs (if interpretability heads are enabled)
            projected_mastery = None
            projected_gains = None
            
            if hasattr(model, 'mastery_head') and model.mastery_head is not None:
                projected_mastery = torch.sigmoid(model.mastery_head(context_seq))
                
            if hasattr(model, 'gain_head') and model.gain_head is not None:
                projected_gains = model.gain_head(value_seq)
            
            # Calculate main loss
            main_loss = compute_loss(predictions, responses)
            total_loss = main_loss
            
            # Add auxiliary losses if interpretability heads exist
            if projected_gains is not None:
                non_negative_loss = torch.mean(torch.relu(-projected_gains))
                total_loss += 0.1 * non_negative_loss  # Configurable weight
                
            if projected_mastery is not None:
                consistency_loss = compute_consistency_loss(projected_mastery, predictions, questions)
                total_loss += 0.1 * consistency_loss  # Configurable weight
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            # MONITOR INTERPRETABILITY DURING TRAINING
            if projected_mastery is not None and projected_gains is not None:
                interp_monitor(
                    batch_idx=batch_idx,
                    context_seq=context_seq,
                    value_seq=value_seq, 
                    projected_mastery=projected_mastery,
                    projected_gains=projected_gains,
                    predictions=predictions,
                    questions=questions,
                    responses=responses
                )
        
        # Epoch-level logging
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

def compute_consistency_loss(projected_mastery, predictions, questions):
    """
    Consistency loss: mastery for target skill should correlate with prediction.
    """
    batch_size, seq_len = questions.shape
    consistency_losses = []
    
    for b in range(batch_size):
        for t in range(seq_len):
            skill_id = questions[b, t].item()
            skill_mastery = projected_mastery[b, t, skill_id]
            prediction = predictions[b, t]
            
            # MSE between skill mastery and prediction for that skill
            loss = (skill_mastery - prediction) ** 2
            consistency_losses.append(loss)
    
    return torch.stack(consistency_losses).mean()

# Usage example:
# enhanced_training_loop(model, train_loader, optimizer, num_epochs=10)