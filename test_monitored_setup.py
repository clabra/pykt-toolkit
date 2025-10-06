#!/usr/bin/env python3
"""
Quick test of GainAKT2Monitored model functionality.

This script tests the monitored model and interpretability monitoring
to ensure everything works before launching full training.
"""

import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

import torch
from pykt.models.gainakt2_monitored import create_monitored_model
from examples.interpretability_monitor import InterpretabilityMonitor

def test_monitored_model():
    """Test that the monitored model works correctly."""
    print("Testing GainAKT2Monitored model...")
    
    # Create a small test configuration
    config = {
        'num_c': 50,  # Small number of concepts for testing
        'seq_len': 10,
        'd_model': 64,
        'n_heads': 4,
        'num_encoder_blocks': 2,
        'd_ff': 128,
        'dropout': 0.1,
        'emb_type': 'qid',
        'non_negative_loss_weight': 0.1,
        'consistency_loss_weight': 0.05,
        'monitor_frequency': 2
    }
    
    # Create model
    model = create_monitored_model(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create interpretability monitor
    monitor = InterpretabilityMonitor(model, log_frequency=2)
    model.set_monitor(monitor)
    print("✓ Interpretability monitor attached")
    
    # Create sample data
    batch_size = 4
    seq_len = 10
    
    questions = torch.randint(0, config['num_c'], (batch_size, seq_len))
    responses = torch.randint(0, 2, (batch_size, seq_len))
    
    print(f"✓ Sample data created: {questions.shape}, {responses.shape}")
    
    # Test standard forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(questions, responses)
        print(f"✓ Standard forward pass: {outputs['predictions'].shape}")
        
        # Test monitored forward pass
        monitored_outputs = model.forward_with_states(questions, responses, batch_idx=0)
        print(f"✓ Monitored forward pass: {monitored_outputs['predictions'].shape}")
        print(f"✓ Interpretability loss: {monitored_outputs['interpretability_loss'].item():.4f}")
        
        # Check that required outputs are present
        required_keys = ['predictions', 'context_seq', 'value_seq', 
                        'projected_mastery', 'projected_gains', 'interpretability_loss']
        for key in required_keys:
            assert key in monitored_outputs, f"Missing key: {key}"
        print("✓ All required outputs present")
        
        # Test interpretability monitoring call
        monitor(
            batch_idx=0,
            context_seq=monitored_outputs['context_seq'],
            value_seq=monitored_outputs['value_seq'],
            projected_mastery=monitored_outputs['projected_mastery'],
            projected_gains=monitored_outputs['projected_gains'],
            predictions=monitored_outputs['predictions'],
            questions=questions,
            responses=responses
        )
        print("✓ Interpretability monitoring executed successfully")
    
    print("✅ All tests passed! Model is ready for training.")
    return True

def test_training_components():
    """Test training-related components."""
    print("\nTesting training components...")
    
    # Test imports
    try:
        import importlib.util
        spec = importlib.util.find_spec("pykt.datasets")
        if spec is not None:
            print("✓ Dataset module available")
        else:
            print("✗ Dataset module not found")
            return False
    except Exception as e:
        print(f"✗ Dataset import test failed: {e}")
        return False
    
    # Test loss function
    try:
        import torch.nn as nn
        _ = nn.BCELoss()
        print("✓ Loss function available")
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False
    
    # Test optimizer
    try:
        dummy_params = torch.nn.Parameter(torch.randn(10, 10))
        _ = torch.optim.Adam([dummy_params], lr=0.001)
        print("✓ Optimizer available")
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        return False
    
    print("✅ Training components ready!")
    return True

def main():
    """Run all tests."""
    print("="*50)
    print("TESTING GAINAKT2 INTERPRETABILITY MONITORING SETUP")
    print("="*50)
    
    try:
        # Test model functionality
        test_monitored_model()
        
        # Test training components  
        test_training_components()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("✅ Ready to launch training with interpretability monitoring")
        print("="*50)
        
        print("\nTo start training, run:")
        print("python /workspaces/pykt-toolkit/launch_monitored_training.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Please fix the issues before launching training.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)