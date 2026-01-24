# Ablation Control Center - Parameter Precedence Summary

## Updated Implementation (Jan 24, 2026)

### Parameter Precedence Rules

**For ablation-related parameters only** (`lambda_sup`, `lambda_ref`, `lambda_probe`, `lambda_initmastery`, `lambda_rate`, `personalization`):

```
ablation mode > command-line > config files > defaults
```

### Key Behaviors

#### 1. `ablation='none'` (Passthrough Mode)
- **No overrides applied**
- Uses standard precedence: command-line > config files > defaults
- Example:
  ```bash
  --ablation none --lambda_ref 0.5 --lambda_probe 0.0
  # Result: Uses 0.5 and 0.0 exactly as specified
  ```

#### 2. Other Ablation Modes (`all`, `reference`, `probe`, `personalization`)
- **Ablation mode overrides everything**
- Command-line parameters are ignored if they conflict with ablation requirements
- Example:
  ```bash
  --ablation all --lambda_ref 0.5
  # Result: lambda_ref is set to 0 (ablation mode wins over 0.5)
  ```

### Quick Reference

| Ablation Mode | Lambda Overrides | Behavior |
|:---|:---|:---|
| `none` | None | Use all parameters as-is |
| `all` | All to 0 (except lambda_sup=1.0) | Pure neural baseline |
| `reference` | lambda_ref=0, lambda_probe=0, etc. | No BKT reference |
| `probe` | lambda_probe=0, etc. | No probe training |
| `personalization` | personalization=False, etc. | No student params |

### Implementation Files

- **Core Logic**: `/home/conchalabra/projects/dl/pykt-toolkit/pykt/models/init_model.py`
  - Function: `validate_and_apply_ablation_config()`
  
- **Documentation**: `/home/conchalabra/projects/dl/pykt-toolkit/assistant/ablation.md`
  - Section 7: Ablation Control Center

- **Tests**: `/home/conchalabra/projects/dl/pykt-toolkit/test_ablation_control_center.py`
  - All tests passing ✅

### Example Output

```
======================================================================
ABLATION CONTROL CENTER
======================================================================
Ablation mode: 'all' (Pure Neural (AKT) - all grounding ablated)
Applying ablation overrides (ablation config takes precedence):

  lambda_sup           = 1.0    ✓ (unchanged: 1.0)
  lambda_ref           = 0      OVERRIDE: 0.5 → 0
  lambda_probe         = 0      OVERRIDE: 1.0 → 0
  lambda_initmastery   = 0      ✓ set
  lambda_rate          = 0      ✓ set
  personalization      = False  ✓ set

Overridden parameters: lambda_ref, lambda_probe
======================================================================
```

This clearly shows when ablation mode overrides user-specified values.
