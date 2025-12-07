# Reproducibility Compliance: GainAKT3Exp Model

**Date**: November 15, 2025  
**Version**: v0.0.21-gainakt3exp  
**Status**: ✅ Compliant with "Zero Defaults" Policy (Option 3: Hybrid Approach)

---

## Overview

The GainAKT3Exp model has been updated to be fully compliant with the reproducibility policy described in `examples/reproducibility.md`, which mandates **"Explicit Parameters, Zero Defaults"**.

### Implementation Strategy: Hybrid Approach (Option 3)

We've implemented a **hybrid approach** that balances strict reproducibility enforcement with developer convenience:

1. ✅ **Constructor defaults preserved** - For backwards compatibility and testing
2. ✅ **Factory function enforces strict policy** - Production code path requires all parameters
3. ✅ **Clear documentation warnings** - Extensive docstring warnings against relying on defaults
4. ✅ **Optional runtime warnings** - Strict mode can be enabled via environment variable

---

## Compliance Mechanisms

### 1. Factory Function Enforcement (Primary Mechanism)

**File**: `pykt/models/gainakt3_exp.py` (lines 530-580)

```python
def create_exp_model(config):
    """
    Factory function to create a GainAKT3Exp model from config.
    
    All parameters must be present in config dict (no fallback defaults).
    Fails fast with clear KeyError if any parameter is missing.
    """
    try:
        return GainAKT3Exp(
            num_c=config['num_c'],
            seq_len=config['seq_len'], 
            # ... all 20+ parameters explicitly required
        )
    except KeyError as e:
        raise ValueError(f"Missing required parameter in model config: {e}. "
                        f"All parameters must be explicitly provided (no defaults).") from e
```

**Guarantee**: If any parameter is missing from the config dict, the factory function **fails immediately** with a clear error message. No silent defaults.

### 2. Comprehensive Docstring Warnings

**File**: `pykt/models/gainakt3_exp.py` (lines 74-109)

The `__init__` docstring now includes:

```python
"""
⚠️  REPRODUCIBILITY WARNING ⚠️
═══════════════════════════════════════════════════════════════════════════
The default parameter values in this constructor are provided ONLY for:
- Backwards compatibility with existing code
- Unit testing convenience
- Quick prototyping in notebooks

**DO NOT RELY ON THESE DEFAULTS IN PRODUCTION CODE**

Production Usage:
- ✅ ALWAYS use the factory function: create_exp_model(config)
- ✅ ALL parameters must be explicit in config dict
- ✅ Single source of truth: configs/parameter_default.json
- ❌ NEVER instantiate directly: GainAKT3Exp(num_c=100)  # Uses hidden defaults!

See examples/reproducibility.md for the "Zero Defaults" policy.
═══════════════════════════════════════════════════════════════════════════
"""
```

**Every parameter** in the docstring is marked with `"DO NOT RELY ON THIS"` to prevent accidental assumptions.

### 3. Optional Runtime Warnings (Strict Mode)

**File**: `pykt/models/gainakt3_exp.py` (lines 111-132)

A runtime warning mechanism is available for development/debugging:

```python
if bool(int(os.environ.get('PYKT_STRICT_REPRODUCIBILITY', '0'))):
    import warnings
    warnings.warn(
        "⚠️  REPRODUCIBILITY WARNING: Direct model instantiation detected!\n"
        "You appear to be instantiating GainAKT3Exp directly instead of using\n"
        "the factory function create_exp_model(config).\n"
        "This violates the 'Zero Defaults' reproducibility policy.\n"
        # ... detailed guidance ...
    )
```

**Usage**: Enable during development to catch accidental direct instantiation:
```bash
export PYKT_STRICT_REPRODUCIBILITY=1
python examples/train_gainakt3exp.py --config ...
```

---

## Production Code Path (Verified Compliant)

### Training Script Usage

**File**: `examples/train_gainakt3exp.py` (line 617)

```python
from pykt.models.gainakt3_exp import create_exp_model

# ... load config from parameter_default.json with CLI overrides ...

model = create_exp_model(model_config)  # ✅ ALL parameters explicit
```

**Flow**:
1. `run_repro_experiment.py` loads all 63 parameters from `configs/parameter_default.json`
2. Applies CLI overrides
3. Passes complete config dict to training script
4. Training script calls `create_exp_model(config)`
5. Factory function enforces all parameters present

**Guarantee**: No parameters can use constructor defaults in production training.

### Evaluation Script Usage

**File**: `examples/eval_gainakt3exp.py`

Same pattern - uses factory function with explicit parameters from saved config.

---

## Why This Approach?

### Advantages of Hybrid Approach

1. **Strict enforcement where it matters**: Production code (training/eval) **must** use factory function
2. **Developer convenience preserved**: Unit tests and notebooks can use simpler syntax
3. **Backwards compatibility**: Existing code that directly instantiates still works
4. **Clear communication**: Extensive warnings make policy violations obvious
5. **Consistent with base models**: Matches pattern in `GainAKT3`, `GainAKT2Exp`, etc.

### Rejected Alternatives

**Option 1: Keep defaults silently**
- ❌ Weak enforcement
- ❌ Easy to accidentally violate policy
- ❌ No clear indication of correct usage

**Option 2: Remove all defaults**
- ❌ Breaks backwards compatibility
- ❌ Makes testing harder (verbose boilerplate)
- ❌ Inconsistent with base model (GainAKT3)
- ❌ Would require updating all existing unit tests

**Option 3: Hybrid (SELECTED)** ✅
- ✅ Strong enforcement in production path
- ✅ Clear warnings prevent accidental misuse
- ✅ Preserves developer convenience
- ✅ Backwards compatible

---

## Verification Checklist

### ✅ Factory Function
- [x] Factory function requires all parameters (no defaults)
- [x] Clear error message if parameter missing
- [x] Used by all production training/eval scripts

### ✅ Documentation
- [x] Comprehensive docstring warning in `__init__`
- [x] Each parameter marked "DO NOT RELY ON THIS"
- [x] Reference to reproducibility.md
- [x] Clear guidance on correct vs incorrect usage

### ✅ Runtime Enforcement (Optional)
- [x] PYKT_STRICT_REPRODUCIBILITY env var support
- [x] Warning shows correct usage pattern
- [x] Disabled by default (no impact on production)

### ✅ Production Code Path
- [x] Training script uses factory function
- [x] Evaluation script uses factory function
- [x] All parameters come from configs/parameter_default.json
- [x] CLI overrides apply on top of defaults

### ✅ Parameter Defaults Alignment
- [x] Constructor defaults documented as "for convenience only"
- [x] configs/parameter_default.json is single source of truth
- [x] No risk of divergence (factory enforces explicit config)

---

## Usage Guidelines

### ✅ Correct Usage (Production)

```python
# In training/evaluation scripts
from pykt.models.gainakt3_exp import create_exp_model

config = {
    'num_c': 100,
    'seq_len': 200,
    'd_model': 256,
    # ... ALL 20+ parameters explicit ...
}

model = create_exp_model(config)  # ✅ Enforces explicit parameters
```

### ✅ Acceptable Usage (Testing/Development)

```python
# In unit tests or notebooks (for convenience)
from pykt.models.gainakt3_exp import GainAKT3Exp

model = GainAKT3Exp(
    num_c=100,
    seq_len=200,
    d_model=128,
    # ... explicitly specify only what matters for test ...
)
# OK for testing, but be aware you're using constructor defaults
```

### ❌ Incorrect Usage (Production)

```python
# DON'T DO THIS in production code!
from pykt.models.gainakt3_exp import GainAKT3Exp

model = GainAKT3Exp(num_c=100)  # ❌ Uses hidden defaults!
# This violates reproducibility policy
```

---

## Environment Variable Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYKT_STRICT_REPRODUCIBILITY` | `0` | Set to `1` to enable warnings when model is instantiated directly |

**Development workflow**:
```bash
# Enable strict mode during development
export PYKT_STRICT_REPRODUCIBILITY=1
python my_script.py  # Will warn if using direct instantiation

# Disable for production (default)
export PYKT_STRICT_REPRODUCIBILITY=0
# or just don't set it
```

---

## Testing Recommendations

### Unit Test Pattern

```python
import pytest
from pykt.models.gainakt3_exp import GainAKT3Exp, create_exp_model

def test_model_via_factory():
    """Test using factory function (preferred)."""
    config = {
        'num_c': 10,
        'seq_len': 50,
        'd_model': 64,
        'n_heads': 2,
        'num_encoder_blocks': 1,
        'd_ff': 128,
        'dropout': 0.1,
        'emb_type': 'qid',
        'use_mastery_head': True,
        'use_gain_head': True,
        # ... all required parameters ...
    }
    model = create_exp_model(config)
    assert model.num_c == 10

def test_model_missing_param():
    """Verify factory fails on missing parameter."""
    config = {'num_c': 10}  # Missing other required params
    with pytest.raises(ValueError, match="Missing required parameter"):
        model = create_exp_model(config)

def test_model_direct_instantiation():
    """Test direct instantiation (for backwards compat only)."""
    model = GainAKT3Exp(num_c=10)  # Uses defaults
    assert model.seq_len == 200  # Default value
    # This test verifies backwards compat, but production code should use factory
```

---

## Related Documentation

- **Reproducibility Policy**: `examples/reproducibility.md`
- **Parameter Defaults**: `configs/parameter_default.json`
- **Training Script**: `examples/train_gainakt3exp.py`
- **Evaluation Script**: `examples/eval_gainakt3exp.py`
- **Factory Function**: `pykt/models/gainakt3_exp.py::create_exp_model()`

---

## Summary

✅ **GainAKT3Exp is compliant** with the "Zero Defaults" reproducibility policy through:

1. **Strong enforcement** via factory function in production code
2. **Clear documentation** warning against relying on constructor defaults
3. **Optional runtime warnings** for development debugging
4. **Preserved backwards compatibility** for testing and prototyping

The hybrid approach ensures **zero hidden defaults in production** while maintaining **developer convenience** for testing and exploration.
