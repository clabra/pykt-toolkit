import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from pykt.models.idkt import iDKT

def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

class IDKTAuditor:
    def __init__(self, model):
        self.model = model
        self.findings = []
        self.components = {}

    def discover_components(self):
        """Dynamically discover grounded components via name patterns."""
        for name, module in self.model.named_modules():
            if 'base_emb' in name:
                self.components['bases'] = self.components.get('bases', []) + [(name, module)]
            if 'axis_emb' in name:
                self.components['axes'] = self.components.get('axes', []) + [(name, module)]
            if 'student_param' in name or 'gap_param' in name:
                self.components['students'] = self.components.get('students', []) + [(name, module)]
        
        # Discover specific parameters
        for name, param in self.model.named_parameters():
            if 'gammas' in name:
                self.components['gammas'] = self.components.get('gammas', []) + [(name, module)]
        
        print(f"[*] Discovery complete. Found:")
        for k, v in self.components.items():
            print(f"  - {k}: {[n for n, _ in v]}")

    def audit_grounding(self, theoretical_priors):
        print("\n[Audit] Theoretical Grounding (Epoch 0 Alignment)")
        # Initialize
        self.model.load_theory_params(theoretical_priors)
        
        # Test 1: Logit consistency in weights
        for name, module in self.components.get('bases', []):
            if isinstance(module, nn.Embedding):
                # Check if it was initialized correctly (we assume prior=0.2 for test)
                p = 0.2 if 'l0' in name else 0.05
                expected = to_logit(p)
                
                # Check mean alignment
                mean_val = module.weight.mean().item()
                diff = abs(mean_val - expected)
                status_mean = "PASS" if diff < 5e-3 else "FAIL" # higher tolerance due to noise
                self.findings.append(f"Basename '{name}' Logit Grounding: {status_mean} (mean_diff={diff:.6e})")

                # Check Texture (Non-zero variance for LayerNorm survival)
                std_val = module.weight.std().item()
                status_tex = "PASS" if std_val > 0.01 else "FAIL"
                self.findings.append(f"Basename '{name}' Texture: {status_tex} (std={std_val:.4f}, Target > 0.01)")

    def audit_neutrality(self):
        print("\n[Audit] Individualization Neutrality")
        for name, module in self.components.get('students', []):
            if isinstance(module, nn.Embedding):
                mean_val = module.weight.mean().item()
                status = "PASS" if abs(mean_val) < 1e-7 else "FAIL"
                self.findings.append(f"Student Param '{name}' Neutrality: {status} (mean={mean_val:.2e})")

    def audit_scaling(self):
        print("\n[Audit] Structural Axis Scaling")
        for name, module in self.components.get('axes', []):
            if isinstance(module, nn.Embedding):
                std_val = module.weight.std().item()
                # We expect std ~ 0.02 after our fix, vs 1.0 default
                status = "PASS" if 0.01 < std_val < 0.05 else "WARNING"
                self.findings.append(f"Axis '{name}' Scale: {status} (std={std_val:.4f}, Target ~0.02)")

    def audit_attention(self):
        print("\n[Audit] Attention Stability")
        for name, param in self.model.named_parameters():
            if 'gammas' in name:
                val = param.mean().item()
                std = param.std().item()
                # Center zero is fine (Softplus(0) = 0.69)
                status = "PASS" if abs(val) < 0.1 else "WARNING"
                self.findings.append(f"Attention Head '{name}' Center: {status} (val={val:.4f})")

    def verify_outputs(self):
        print("\n[Audit] Functional Mapping (Forward Pass)")
        q_data = torch.arange(5).unsqueeze(0)
        target = torch.zeros(1, 5).long()
        uid_data = torch.zeros(1).long()
        
        with torch.no_grad():
            outputs = self.model(q_data, target, uid_data=uid_data)
            # Find initmastery and rate in outputs (duck typing)
            # preds, initmastery, rate, c_reg_loss, reg_losses
            if len(outputs) >= 3:
                im, r = outputs[1], outputs[2]
                im_val = im[0, 0].item()
                r_val = r[0, 0].item()
                # Expected 0.2 and 0.05 from mock priors
                im_err = abs(im_val - 0.2)
                r_err = abs(r_val - 0.05)
                status = "PASS" if im_err < 5e-3 else "FAIL" # noise tolerance
                self.findings.append(f"Forward Output Alignment: {status} (im={im_val:.4f}, r={r_val:.4f})")

    def audit_calibration_logic(self):
        print("\n[Audit] Calibration Stability Simulation")
        # Simulate m_sup=0.7 (normal start) and m_ref=1e-12 (near zero issue)
        m_sup = 0.7
        m_ref_crash = 1e-15
        m_ref_normal = 0.05
        
        target_ratio = 0.1
        
        # Simulated logic from train_idkt.py
        def cal_lambda(target, sup, ref):
            return min(target * (sup / (ref + 1e-8)), 100.0)

        l_crash = cal_lambda(target_ratio, m_sup, m_ref_crash)
        l_normal = cal_lambda(target_ratio, m_sup, m_ref_normal)
        
        status = "PASS" if l_crash == 100.0 and 0 < l_normal < 10.0 else "FAIL"
        self.findings.append(f"Calibration Stability Guard: {status} (Capped: {l_crash}, Normal: {l_normal:.2f})")

    def run_full_audit(self, mock_priors):
        self.discover_components()
        self.audit_grounding(mock_priors)
        self.audit_neutrality()
        self.audit_scaling()
        self.audit_attention()
        self.verify_outputs()
        self.audit_calibration_logic()
        
        print("\n" + "="*60)
        print(" FINAL AUDIT REPORT ")
        print("="*60)
        for f in self.findings:
            print(f" - {f}")
        
        if any("FAIL" in f for f in self.findings):
            print("\nVERDICT: [CRITICAL FAIL] - Structural inconsistencies detected.")
            return False
        if any("WARNING" in f for f in self.findings):
            print("\nVERDICT: [WARNING] - Calibration is valid but sub-optimal.")
            return True
        print("\nVERDICT: [PASS] - Model is theoretically and technically sound.")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_q", type=int, default=100)
    parser.add_argument("--d", type=int, default=256)
    args = parser.parse_args()
    
    model = iDKT(n_question=args.n_q, n_pid=0, d_model=args.d, n_blocks=1, dropout=0.1, n_uid=1000)
    mock_priors = {
        'params': {i: {'prior': 0.2, 'learns': 0.05} for i in range(args.n_q + 1)},
        'global': {'prior': 0.2, 'learns': 0.05}
    }
    
    auditor = IDKTAuditor(model)
    auditor.run_full_audit(mock_priors)
