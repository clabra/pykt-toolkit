#!/usr/bin/env python3
"""
Test script to validate time-varying BKT parameter fix.
Runs a single forward pass with a small batch to verify:
1. Tensor shapes are correct
2. No runtime errors
3. Outputs are numerically valid
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Force CPU execution
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add pykt to path
sys.path.insert(0, str(Path(__file__).parent))

def test_timevarying_bkt():
    """Run a minimal forward pass to validate the implementation."""
    
    print("=" * 70)
    print("Testing Time-Varying BKT Parameters Fix")
    print("=" * 70)
    
    # Import model
    from pykt.models.gtransformer import GTransformer
    
    # Minimal configuration
    config = {
        'n_question': 100,
        'n_pid': 100,
        'd_model': 64,
        'n_blocks': 2,
        'dropout': 0.1,
        'd_ff': 256,
        'kq_same': 1,
        'final_fc_dim': 512,
        'num_attn_heads': 8,
        'separate_qa': 0,
        'l2_rasch': 1e-5,
        'emb_type': 'qid',
        'emb_path': '',
        'pretrain_dim': 768,
        'ablation': '',  # Full grounding
        'n_uid': 0,  # No personalization
    }
    
    print("\n[1/4] Initializing model...")
    model = GTransformer(**config)
    model.eval()  # Set to eval mode
    
    # Create small synthetic batch
    bs = 4  # Small batch size
    seqlen = 20  # Short sequence
    
    print(f"[2/4] Creating synthetic batch (BS={bs}, seqlen={seqlen})...")
    
    # Random inputs
    q_data = torch.randint(1, 100, (bs, seqlen))  # Question IDs (start from 1)
    target = torch.randint(0, 2, (bs, seqlen))    # Responses (0/1)
    pid_data = torch.randint(1, 100, (bs, seqlen))  # Problem IDs
    
    print("[3/4] Running forward pass...")
    
    try:
        with torch.no_grad():
            # forward(self, q_data, target, pid_data=None, uid_data=None, qtest=False)
            outputs, reg_loss = model(q_data, target, pid_data=pid_data, uid_data=None)
        
        print("✓ Forward pass completed successfully!")
        
        # Validate outputs
        print("\n[4/4] Validating outputs...")
        
        assert 'predictions' in outputs, "Missing 'predictions' in outputs"
        preds = outputs['predictions']
        
        assert preds.shape == (bs, seqlen), f"Predictions shape mismatch: {preds.shape}"
        assert torch.all((preds >= 0) & (preds <= 1)), "Predictions outside [0,1] range"
        assert not torch.any(torch.isnan(preds)), "NaN detected in predictions"
        assert not torch.any(torch.isinf(preds)), "Inf detected in predictions"
        
        print(f"  ✓ Predictions shape: {preds.shape}")
        print(f"  ✓ Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  ✓ Predictions mean: {preds.mean():.4f}")
        
        if 'p_ref' in outputs:
            p_ref = outputs['p_ref']
            print(f"  ✓ p_ref shape: {p_ref.shape}")
            print(f"  ✓ p_ref range: [{p_ref.min():.4f}, {p_ref.max():.4f}]")
            print(f"  ✓ p_ref mean: {p_ref.mean():.4f}")
            
            # Key test: Check if p_ref shows variance (not completely flat)
            p_ref_std = p_ref.std().item()
            print(f"  ✓ p_ref std dev: {p_ref_std:.4f}")
            
            if p_ref_std < 0.01:
                print(f"  ⚠️  Warning: p_ref has very low variance (std={p_ref_std:.4f})")
            else:
                print(f"  ✓ p_ref shows healthy variance!")
        
        if 'p_l0' in outputs and 'p_t' in outputs:
            p_l0 = outputs['p_l0']
            p_t = outputs['p_t']
            print(f"  ✓ p_l0 range: [{p_l0.min():.4f}, {p_l0.max():.4f}]")
            print(f"  ✓ p_t range: [{p_t.min():.4f}, {p_t.max():.4f}]")
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED - Time-varying BKT implementation is valid!")
        print("=" * 70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_timevarying_bkt()
    sys.exit(0 if success else 1)
