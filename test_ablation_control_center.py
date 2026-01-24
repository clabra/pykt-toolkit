#!/usr/bin/env python3
"""
Test script for the updated Ablation Control Center behavior.

Tests the parameter precedence: ablation mode > command-line > config files
For ablation-related parameters only (lambda_*, personalization).

Tests:
1. ablation='none' should use parameters as-is (no overrides)
2. ablation='all' should override parameters (ablation mode wins)
3. Other ablation modes should override specific parameters (ablation mode wins)
"""

import sys
import copy

# Add pykt to path
sys.path.insert(0, '/home/conchalabra/projects/dl/pykt-toolkit')

from pykt.models.init_model import validate_and_apply_ablation_config

def test_ablation_none():
    """Test that ablation='none' uses parameters as configured without overrides."""
    print("\n" + "="*70)
    print("TEST 1: ablation='none' (should use parameters as-is)")
    print("="*70)
    
    config = {
        "ablation": "none",
        "lambda_sup": 1.0,
        "lambda_ref": 0.5,
        "lambda_probe": 0.0,
        "lambda_initmastery": 0.1,
        "lambda_rate": 0.1,
        "personalization": False
    }
    
    original_config = copy.deepcopy(config)
    result = validate_and_apply_ablation_config(config, source="test")
    
    # Verify no parameters were changed
    assert result["lambda_ref"] == 0.5, f"lambda_ref should be 0.5, got {result['lambda_ref']}"
    assert result["lambda_probe"] == 0.0, f"lambda_probe should be 0.0, got {result['lambda_probe']}"
    assert result["lambda_initmastery"] == 0.1, f"lambda_initmastery should be 0.1, got {result['lambda_initmastery']}"
    
    print("✅ PASSED: Parameters were not overridden")
    return True

def test_ablation_all():
    """Test that ablation='all' overrides parameters correctly."""
    print("\n" + "="*70)
    print("TEST 2: ablation='all' (should override parameters)")
    print("="*70)
    
    config = {
        "ablation": "all",
        "lambda_sup": 1.0,
        "lambda_ref": 0.5,  # Should be overridden to 0
        "lambda_probe": 1.0,  # Should be overridden to 0
        "lambda_initmastery": 0.1,  # Should be overridden to 0
        "lambda_rate": 0.1,  # Should be overridden to 0
        "personalization": True  # Should be overridden to False
    }
    
    result = validate_and_apply_ablation_config(config, source="test")
    
    # Verify parameters were overridden
    assert result["lambda_ref"] == 0, f"lambda_ref should be 0, got {result['lambda_ref']}"
    assert result["lambda_probe"] == 0, f"lambda_probe should be 0, got {result['lambda_probe']}"
    assert result["lambda_initmastery"] == 0, f"lambda_initmastery should be 0, got {result['lambda_initmastery']}"
    assert result["lambda_rate"] == 0, f"lambda_rate should be 0, got {result['lambda_rate']}"
    assert result["personalization"] == False, f"personalization should be False, got {result['personalization']}"
    
    print("✅ PASSED: Parameters were correctly overridden")
    return True

def test_ablation_reference():
    """Test that ablation='reference' overrides correctly."""
    print("\n" + "="*70)
    print("TEST 3: ablation='reference' (should ablate reference pipeline)")
    print("="*70)
    
    config = {
        "ablation": "reference",
        "lambda_sup": 1.0,
        "lambda_ref": 1.0,  # Should be overridden to 0
        "lambda_probe": 1.0,  # Should be overridden to 0
        "lambda_initmastery": 0.1,  # Should be overridden to 0
        "lambda_rate": 0.1,  # Should be overridden to 0
        "personalization": False
    }
    
    result = validate_and_apply_ablation_config(config, source="test")
    
    # Verify correct overrides
    assert result["lambda_ref"] == 0, f"lambda_ref should be 0, got {result['lambda_ref']}"
    assert result["lambda_probe"] == 0, f"lambda_probe should be 0, got {result['lambda_probe']}"
    assert result["lambda_initmastery"] == 0, f"lambda_initmastery should be 0, got {result['lambda_initmastery']}"
    assert result["lambda_rate"] == 0, f"lambda_rate should be 0, got {result['lambda_rate']}"
    
    print("✅ PASSED: Reference pipeline correctly ablated")
    return True

def test_ablation_probe():
    """Test that ablation='probe' overrides correctly."""
    print("\n" + "="*70)
    print("TEST 4: ablation='probe' (should ablate probe loss)")
    print("="*70)
    
    config = {
        "ablation": "probe",
        "lambda_sup": 1.0,
        "lambda_ref": 1.0,
        "lambda_probe": 1.0,  # Should be overridden to 0
        "lambda_initmastery": 0.1,  # Should be overridden to 0
        "lambda_rate": 0.1,  # Should be overridden to 0
        "personalization": False
    }
    
    result = validate_and_apply_ablation_config(config, source="test")
    
    # Verify correct overrides
    assert result["lambda_ref"] == 1.0, f"lambda_ref should be 1.0, got {result['lambda_ref']}"
    assert result["lambda_probe"] == 0, f"lambda_probe should be 0, got {result['lambda_probe']}"
    assert result["lambda_initmastery"] == 0, f"lambda_initmastery should be 0, got {result['lambda_initmastery']}"
    assert result["lambda_rate"] == 0, f"lambda_rate should be 0, got {result['lambda_rate']}"
    
    print("✅ PASSED: Probe loss correctly ablated")
    return True

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ABLATION CONTROL CENTER - UPDATED BEHAVIOR TESTS")
    print("="*70)
    
    tests = [
        test_ablation_none,
        test_ablation_all,
        test_ablation_reference,
        test_ablation_probe,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
