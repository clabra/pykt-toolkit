#!/usr/bin/env python3
"""
Test script to verify wandb sweep setup without actually running training.
"""

import yaml
import sys

# Add the project root to Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

def test_sweep_config():
    """Test that the sweep configuration is valid."""
    print("Testing sweep configuration...")
    
    # Load and validate sweep config
    try:
        with open('sweep_config_gainakt2.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Sweep configuration loaded successfully")
        print(f"  Method: {config['method']}")
        print(f"  Metric: {config['metric']['name']} ({config['metric']['goal']})")
        print(f"  Parameters: {len(config['parameters'])}")
        
        # Check parameter ranges
        params = config['parameters']
        
        # Validate d_model and n_heads compatibility
        d_models = params['d_model']['values']
        n_heads = params['n_heads']['values']
        
        print("\n✓ Testing parameter compatibility...")
        valid_combinations = []
        for d_model in d_models:
            for n_head in n_heads:
                if d_model % n_head == 0:
                    valid_combinations.append((d_model, n_head))
        
        print(f"  Valid (d_model, n_heads) combinations: {len(valid_combinations)}")
        for combo in valid_combinations:
            print(f"    d_model={combo[0]}, n_heads={combo[1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading sweep config: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported."""
    print("\nTesting imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import wandb
        print(f"✓ wandb {wandb.__version__}")
        
        from pykt.models.gainakt2_monitored import create_monitored_model
        print("✓ GainAKT2Monitored model")
        
        from examples.interpretability_monitor import InterpretabilityMonitor
        print("✓ InterpretabilityMonitor")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation with sample parameters."""
    print("\nTesting model creation...")
    
    try:
        # Test with smallest configuration
        model_config = {
            'num_c': 100,
            'seq_len': 200,
            'd_model': 128,
            'n_heads': 4,
            'num_encoder_blocks': 2,
            'd_ff': 256,
            'dropout': 0.1,
            'emb_type': 'qid',
            'non_negative_loss_weight': 0.1,
            'consistency_loss_weight': 0.05,
            'monitor_frequency': 25,
            'optimizer': 'adam'
        }
        
        from pykt.models.gainakt2_monitored import create_monitored_model
        from examples.interpretability_monitor import InterpretabilityMonitor
        
        model = create_monitored_model(model_config)
        monitor = InterpretabilityMonitor(model, log_frequency=25)
        model.set_monitor(monitor)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully with {params} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("WANDB SWEEP SETUP VALIDATION")
    print("="*60)
    
    tests = [
        test_sweep_config,
        test_imports,
        test_model_creation
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    if all(results):
        print("✅ ALL TESTS PASSED!")
        print("\nYour wandb sweep setup is ready. To start:")
        print("1. Log into wandb: wandb login")
        print("2. Initialize sweep: python initialize_sweep.py")
        print("3. Or run directly: wandb sweep sweep_config_gainakt2.yaml")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)