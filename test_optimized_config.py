#!/usr/bin/env python3
"""
Test script to verify the optimized hyperparameters are correctly applied.
"""

import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

def test_optimized_config():
    """Test that the optimized configuration is correctly set."""
    
    print("="*60)
    print("TESTING OPTIMIZED HYPERPARAMETER CONFIGURATION")
    print("="*60)
    
    # Import the main training script
    import quick_launch_monitored
    
    # We'll examine the configuration by reading the file
    with open('/workspaces/pykt-toolkit/quick_launch_monitored.py', 'r') as f:
        content = f.read()
    
    # Expected optimized values
    expected_values = {
        'd_model': 512,
        'n_heads': 4,
        'num_encoder_blocks': 4,
        'd_ff': 512,
        'batch_size': 32,
        'dropout': 0.311646,
        'non_negative_loss_weight': 0.485828,
        'consistency_loss_weight': 0.173548,
        'learning_rate': 0.000103,
        'weight_decay': 0.000276
    }
    
    print("✓ Checking optimized parameter values in script:")
    
    # Check model architecture parameters
    checks = {
        'd_model': "'d_model': 512" in content,
        'n_heads': "'n_heads': 4" in content,
        'num_encoder_blocks': "'num_encoder_blocks': 4" in content,
        'd_ff': "'d_ff': 512" in content,
        'dropout': "'dropout': 0.311646" in content,
        'batch_size': "batch_size = 32" in content,
        'non_negative_loss_weight': "'non_negative_loss_weight': 0.485828" in content,
        'consistency_loss_weight': "'consistency_loss_weight': 0.173548" in content,
        'learning_rate': "lr=0.000103" in content,
        'weight_decay': "weight_decay=0.000276" in content
    }
    
    all_passed = True
    for param, check in checks.items():
        status = "✓" if check else "✗"
        print(f"  {status} {param}: {expected_values[param]}")
        if not check:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL OPTIMIZED PARAMETERS CORRECTLY APPLIED!")
        print("📈 Expected AUC improvement: 0.7245 → 0.7250 (+0.0005)")
        print("🚀 Ready to run optimized training!")
    else:
        print("❌ Some parameters not correctly set. Please check the configuration.")
    
    print("="*60)
    
    return all_passed

def test_model_creation():
    """Test that the model can be created with optimized parameters."""
    
    print("\nTesting model creation with optimized parameters...")
    
    try:
        from pykt.models.gainakt2_monitored import create_monitored_model
        from examples.interpretability_monitor import InterpretabilityMonitor
        
        # Test configuration
        model_config = {
            'num_c': 100,
            'seq_len': 200,
            'd_model': 512,
            'n_heads': 4,
            'num_encoder_blocks': 4,
            'd_ff': 512,
            'dropout': 0.311646,
            'emb_type': 'qid',
            'non_negative_loss_weight': 0.485828,
            'consistency_loss_weight': 0.173548,
            'monitor_frequency': 25,
            'optimizer': 'adam'
        }
        
        model = create_monitored_model(model_config)
        monitor = InterpretabilityMonitor(model, log_frequency=25)
        model.set_monitor(monitor)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully with {params:,} parameters")
        print(f"✓ Optimized architecture: d_model={model_config['d_model']}, n_heads={model_config['n_heads']}")
        print(f"✓ Interpretability weights: neg_loss={model_config['non_negative_loss_weight']:.3f}, consistency={model_config['consistency_loss_weight']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    """Run all tests."""
    
    config_test = test_optimized_config()
    model_test = test_model_creation()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    if config_test and model_test:
        print("🎯 SUCCESS: Optimized configuration is ready!")
        print("\nTo run with optimized parameters:")
        print("  python quick_launch_monitored.py")
        print("\nExpected improvements:")
        print("  • Better AUC: 0.7250 vs 0.7245 baseline")
        print("  • Optimized interpretability constraint weights") 
        print("  • More efficient architecture (d_model=512, n_heads=4)")
        return True
    else:
        print("❌ FAILED: Issues found with configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)