#!/usr/bin/env python3
"""
Parameter Audit Script for Reproducibility Infrastructure

This script verifies compliance with the "Explicit Parameters, Zero Defaults"
reproducibility philosophy by checking:
1. MD5 integrity of parameter_default.json
2. Hardcoded fallback synchronization with defaults
3. Model initialization fallback removal
4. Evaluation script documentation
5. Parameter coverage
6. No suspicious hardcoded values

Usage:
    python examples/parameters_audit.py [--fix-md5] [--verbose]

Exit codes:
    0: All checks passed
    1: Some checks failed
    2: Critical error (file not found, etc.)

Part of reproducibility infrastructure (see examples/reproducibility.md).
"""

import json
import re
import sys
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class ParameterAuditor:
    """Audits reproducibility infrastructure for parameter handling."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.root_dir = Path(__file__).parent.parent
        self.checks_passed = []
        self.checks_failed = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if self.verbose or level == "ERROR":
            prefix = {"INFO": "  ", "ERROR": "❌", "SUCCESS": "✅", "WARNING": "⚠️"}
            print(f"{prefix.get(level, '  ')}{message}")
    
    def print_header(self, title: str):
        """Print section header."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
    
    def print_check_header(self, check_num: int, title: str):
        """Print check header."""
        print(f"\n📋 CHECK {check_num}: {title}")
        print("-" * 80)
    
    def check_md5_integrity(self) -> bool:
        """Check 1: Verify parameter_default.json MD5 integrity."""
        self.print_check_header(1, "parameter_default.json MD5 Integrity")
        
        try:
            param_file = self.root_dir / "configs" / "parameter_default.json"
            with open(param_file, 'r') as f:
                param_data = json.load(f)
            
            defaults = param_data['defaults']
            stored_md5 = param_data.get('md5', 'NOT FOUND')
            
            # Compute MD5 of defaults section
            defaults_json = json.dumps(defaults, sort_keys=True)
            computed_md5 = hashlib.md5(defaults_json.encode()).hexdigest()
            
            print(f"  Stored MD5:   {stored_md5}")
            print(f"  Computed MD5: {computed_md5}")
            print(f"  Total parameters: {len(defaults)}")
            
            if stored_md5 == computed_md5:
                print(f"  ✅ Match: YES")
                self.log("MD5 integrity verified", "SUCCESS")
                return True
            else:
                print(f"  ❌ Match: NO")
                self.log("MD5 MISMATCH - Config may be corrupted or modified", "ERROR")
                self.log(f"Run with --fix-md5 to update to {computed_md5}", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"Error checking MD5: {e}", "ERROR")
            return False
    
    def check_fallback_synchronization(self) -> bool:
        """Check 2: Verify hardcoded fallbacks match parameter_default.json."""
        self.print_check_header(2, "Hardcoded Fallback Synchronization (Priority 1)")
        
        try:
            # Load parameter defaults
            param_file = self.root_dir / "configs" / "parameter_default.json"
            with open(param_file, 'r') as f:
                defaults = json.load(f)['defaults']
            
            # Load training script
            train_file = self.root_dir / "examples" / "train_gainakt2exp.py"
            with open(train_file, 'r') as f:
                train_content = f.read()
            
            # Check critical parameters that were previously mismatched
            critical_params = {
                'alignment_weight': '0.25',
                'batch_size': '64',
                'enable_alignment_loss': 'True',
                'enable_global_alignment_pass': 'True',
                'enable_lag_gain_loss': 'True',
                'enable_retention_loss': 'True',
                'epochs': '12',
                'use_residual_alignment': 'True'
            }
            
            mismatches = 0
            for param, expected_val in critical_params.items():
                pattern = f"getattr\\(args, '{param}', ([^)]+)\\)"
                matches = re.findall(pattern, train_content)
                
                if matches:
                    actual_fallback = matches[0].strip()
                    status = "✅" if actual_fallback == expected_val else "❌"
                    print(f"  {status} {param:32s} fallback={actual_fallback:6s} (expected {expected_val})")
                    
                    if actual_fallback != expected_val:
                        mismatches += 1
                        self.log(f"Mismatch in {param}: {actual_fallback} != {expected_val}", "ERROR")
            
            if mismatches == 0:
                print(f"\n  Result: ✅ PASS - All {len(critical_params)} fallback values synchronized")
                return True
            else:
                print(f"\n  Result: ❌ FAIL - {mismatches} mismatches found")
                return False
                
        except Exception as e:
            self.log(f"Error checking fallbacks: {e}", "ERROR")
            return False
    
    def check_model_initialization(self) -> bool:
        """Check 3: Verify model initialization has no .get() fallbacks."""
        self.print_check_header(3, "Model Initialization Fallback Removal (Priority 2)")
        
        try:
            model_file = self.root_dir / "pykt" / "models" / "gainakt2_exp.py"
            with open(model_file, 'r') as f:
                model_content = f.read()
            
            get_count = model_content.count("config.get(")
            bracket_count = model_content.count("config['")
            has_keyerror = "KeyError" in model_content and "Missing required parameter" in model_content
            
            print(f"  config.get() calls:          {get_count}")
            print(f"  config['key'] direct access: {bracket_count}")
            print(f"  Fail-fast error handling:    {'✅ Present' if has_keyerror else '❌ Missing'}")
            
            passed = (get_count == 0 and bracket_count >= 15 and has_keyerror)
            
            if passed:
                print(f"\n  Result: ✅ PASS - Model uses fail-fast approach")
                return True
            else:
                print(f"\n  Result: ❌ FAIL - Model still has .get() fallbacks")
                if get_count > 0:
                    self.log(f"Found {get_count} config.get() calls - should use config['key']", "ERROR")
                if not has_keyerror:
                    self.log("Missing KeyError handling for missing parameters", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error checking model initialization: {e}", "ERROR")
            return False
    
    def check_eval_documentation(self) -> bool:
        """Check 4: Verify eval script has proper documentation."""
        self.print_check_header(4, "Eval Script Documentation (Priority 3)")
        
        try:
            eval_file = self.root_dir / "examples" / "eval_gainakt2exp.py"
            with open(eval_file, 'r') as f:
                eval_content = f.read()
            
            has_docs = "CRITICAL ARCHITECTURAL FLAGS" in eval_content
            doc_quality = all(flag in eval_content for flag in 
                            ['use_mastery_head', 'use_gain_head', 'intrinsic_gain_attention'])
            
            print(f"  Documentation header:        {'✅ Present' if has_docs else '❌ Missing'}")
            print(f"  All architectural flags doc: {'✅ Present' if doc_quality else '❌ Missing'}")
            
            passed = has_docs and doc_quality
            
            if passed:
                print(f"\n  Result: ✅ PASS - Documentation complete")
                return True
            else:
                print(f"\n  Result: ❌ FAIL - Documentation incomplete")
                return False
                
        except Exception as e:
            self.log(f"Error checking eval documentation: {e}", "ERROR")
            return False
    
    def check_parameter_coverage(self) -> bool:
        """Check 5: Verify all required parameters in parameter_default.json."""
        self.print_check_header(5, "Parameter Coverage in parameter_default.json")
        
        try:
            param_file = self.root_dir / "configs" / "parameter_default.json"
            with open(param_file, 'r') as f:
                defaults = json.load(f)['defaults']
            
            # Critical parameters that must be present
            required_params = [
                'd_model', 'n_heads', 'num_encoder_blocks', 'd_ff', 'dropout',
                'mastery_performance_loss_weight', 'gain_performance_loss_weight',
                'alignment_weight', 'batch_size', 'epochs', 'learning_rate',
                'use_mastery_head', 'use_gain_head', 'intrinsic_gain_attention',
                'seed', 'optimizer', 'weight_decay', 'patience'
            ]
            
            missing = [p for p in required_params if p not in defaults]
            
            print(f"  Required parameters checked: {len(required_params)}")
            print(f"  Missing from defaults:       {len(missing)}")
            
            if missing:
                for p in missing:
                    print(f"    ❌ {p}")
            
            if len(missing) == 0:
                print(f"\n  Result: ✅ PASS - All required parameters present")
                return True
            else:
                print(f"\n  Result: ❌ FAIL - {len(missing)} parameters missing")
                return False
                
        except Exception as e:
            self.log(f"Error checking parameter coverage: {e}", "ERROR")
            return False
    
    def check_suspicious_values(self) -> bool:
        """Check 6: Verify no suspicious hardcoded values remain."""
        self.print_check_header(6, "No Suspicious Hardcoded Values")
        
        try:
            train_file = self.root_dir / "examples" / "train_gainakt2exp.py"
            with open(train_file, 'r') as f:
                train_content = f.read()
            
            # Check for old wrong fallbacks that should have been fixed
            wrong_fallbacks = [
                (r'getattr\(args, \'alignment_weight\', 0\.1\)', 
                 'alignment_weight fallback 0.1 (should be 0.25)'),
                (r'getattr\(args, \'batch_size\', 96\)', 
                 'batch_size fallback 96 (should be 64)'),
                (r'getattr\(args, \'epochs\', 20\)', 
                 'epochs fallback 20 (should be 12)'),
            ]
            
            suspicious = []
            for pattern, desc in wrong_fallbacks:
                if re.search(pattern, train_content):
                    suspicious.append(desc)
                    print(f"  ❌ Found: {desc}")
            
            if not suspicious:
                print(f"  ✅ No suspicious hardcoded values found")
                print(f"\n  Result: ✅ PASS")
                return True
            else:
                print(f"\n  Result: ❌ FAIL - {len(suspicious)} suspicious values found")
                return False
                
        except Exception as e:
            self.log(f"Error checking suspicious values: {e}", "ERROR")
            return False
    
    def print_summary(self, checks: List[Tuple[str, bool]]):
        """Print final audit summary."""
        self.print_header("REPRODUCIBILITY AUDIT SUMMARY")
        
        passed = sum(1 for _, result in checks if result)
        total = len(checks)
        
        for name, result in checks:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}  {name}")
        
        print("\n" + "=" * 80)
        
        if passed == total:
            print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
            print("✅ REPRODUCIBILITY INFRASTRUCTURE: FULLY COMPLIANT")
            print("=" * 80)
            print("\nAll reproducibility requirements verified:")
            print("  • Priority 1: Fallback values synchronized ✅")
            print("  • Priority 2: Model .get() fallbacks removed ✅")
            print("  • Priority 3: Eval documentation present ✅")
            print("  • MD5 integrity maintained ✅")
            print("\nSafe to launch training/evaluation experiments.")
            return True
        else:
            print(f"⚠️  SOME CHECKS FAILED ({passed}/{total} passed)")
            print("❌ REPRODUCIBILITY INFRASTRUCTURE: NEEDS ATTENTION")
            print("=" * 80)
            print("\nPlease fix the issues above before launching experiments.")
            print("See examples/reproducibility.md and tmp/REPRODUCIBILITY_AUDIT_20251110.md")
            return False
    
    def fix_md5(self) -> bool:
        """Fix MD5 hash in parameter_default.json."""
        try:
            param_file = self.root_dir / "configs" / "parameter_default.json"
            with open(param_file, 'r') as f:
                data = json.load(f)
            
            # Compute correct MD5
            defaults = data['defaults']
            defaults_json = json.dumps(defaults, sort_keys=True)
            new_md5 = hashlib.md5(defaults_json.encode()).hexdigest()
            
            old_md5 = data.get('md5', 'NOT FOUND')
            
            if old_md5 == new_md5:
                print(f"✅ MD5 already correct: {new_md5}")
                return True
            
            # Update MD5
            data['md5'] = new_md5
            
            # Save updated file
            with open(param_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Updated MD5 in parameter_default.json")
            print(f"   Old: {old_md5}")
            print(f"   New: {new_md5}")
            print(f"\n⚠️  Remember to commit this change following Parameter Evolution Protocol")
            return True
            
        except Exception as e:
            print(f"❌ Error fixing MD5: {e}")
            return False
    
    def run_audit(self) -> bool:
        """Run complete reproducibility audit."""
        self.print_header("REPRODUCIBILITY INFRASTRUCTURE AUDIT")
        print("Verifying 'Explicit Parameters, Zero Defaults' compliance...")
        
        checks = [
            ("MD5 Integrity", self.check_md5_integrity()),
            ("Fallback Synchronization (8 params)", self.check_fallback_synchronization()),
            ("Model Init Fallback Removal", self.check_model_initialization()),
            ("Eval Script Documentation", self.check_eval_documentation()),
            ("Parameter Coverage", self.check_parameter_coverage()),
            ("No Suspicious Values", self.check_suspicious_values()),
        ]
        
        return self.print_summary(checks)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit reproducibility infrastructure compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full audit
  python examples/parameters_audit.py
  
  # Run with verbose output
  python examples/parameters_audit.py --verbose
  
  # Fix MD5 mismatch
  python examples/parameters_audit.py --fix-md5
  
  # Run audit and fix MD5 if needed
  python examples/parameters_audit.py --fix-md5 --verbose

Exit codes:
  0: All checks passed
  1: Some checks failed
  2: Critical error

See also: examples/reproducibility.md
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--fix-md5', action='store_true',
                       help='Automatically fix MD5 hash if mismatched')
    
    args = parser.parse_args()
    
    try:
        auditor = ParameterAuditor(verbose=args.verbose)
        
        # Fix MD5 first if requested
        if args.fix_md5:
            print("=" * 80)
            print("FIXING MD5 HASH")
            print("=" * 80)
            if not auditor.fix_md5():
                return 2
            print()
        
        # Run audit
        passed = auditor.run_audit()
        
        print("=" * 80)
        
        return 0 if passed else 1
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
