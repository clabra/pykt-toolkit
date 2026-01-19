
import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from examples.validation.validation_helpers import load_model_from_dir
from pykt.datasets import init_test_datasets

def collect_envelope_data(model, loader, device):
    """
    Collect comprehensive envelope and prediction data.
    """
    
    all_envelope = []
    all_psup = []
    all_pref = []
    all_correct = []
    all_p_l0 = []
    all_p_t = []
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            q = data["qseqs"].long().to(device)
            c = data["cseqs"].long().to(device)
            r = data["rseqs"].long().to(device)
            qshft = data["shft_qseqs"].long().to(device)
            cshft = data["shft_cseqs"].long().to(device)
            rshft = data["shft_rseqs"].long().to(device)
            m = data["masks"].bool().to(device)
            sm = data["smasks"].bool().to(device)
            uids = data.get("uids", None)
            if uids is not None: uids = uids.long().to(device)
            
            # Concatenate sequences
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)
            
            outputs, _ = model(cc.long(), cr.long(), pid_data=cq.long(), uid_data=uids)
            
            # Extract predictions
            p_l0 = outputs['p_l0'][:,1:].cpu().numpy()
            p_t = outputs['p_t'][:,1:].cpu().numpy()
            preds = outputs['predictions'][:,1:].cpu().numpy()
            ref_preds = outputs['reference_preds'][:,1:].cpu().numpy()
            
            for b in range(c.shape[0]):
                m_b = sm[b].cpu().numpy()
                cur_r = rshft[b].cpu().numpy()[m_b]
                cur_p_l0 = p_l0[b][m_b]
                cur_p_t = p_t[b][m_b]
                cur_preds = preds[b][m_b]
                cur_ref_preds = ref_preds[b][m_b]
                
                envelope = np.abs(cur_preds - cur_ref_preds)
                
                all_envelope.extend(envelope)
                all_psup.extend(cur_preds)
                all_pref.extend(cur_ref_preds)
                all_correct.extend(cur_r)
                all_p_l0.extend(cur_p_l0)
                all_p_t.extend(cur_p_t)
            
            if i > 100:  # Sufficient data
                break
    
    return {
        'envelope': np.array(all_envelope),
        'psup': np.array(all_psup),
        'pref': np.array(all_pref),
        'correct': np.array(all_correct),
        'p_l0': np.array(all_p_l0),
        'p_t': np.array(all_p_t),
    }

def plot_envelope_distribution(data, output_dir):
    """
    Generate comprehensive envelope distribution visualizations.
    """
    
    envelope = data['envelope']
    psup = data['psup']
    pref = data['pref']
    correct = data['correct']
    bias = psup - pref
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Main Histogram: Envelope Width Distribution
    ax1 = plt.subplot(3, 3, (1, 4))
    counts, bins, patches = ax1.hist(envelope, bins=100, alpha=0.7, color='royalblue', 
                                      edgecolor='black', linewidth=0.5, density=True)
    
    # Add statistics lines
    mean_env = np.mean(envelope)
    median_env = np.median(envelope)
    p90_env = np.percentile(envelope, 90)
    p95_env = np.percentile(envelope, 95)
    
    ax1.axvline(mean_env, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_env:.3f}')
    ax1.axvline(median_env, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_env:.3f}')
    ax1.axvline(p90_env, color='orange', linestyle=':', linewidth=2, 
                label=f'90th %ile: {p90_env:.3f}')
    
    # Fit kernel density
    kde = stats.gaussian_kde(envelope)
    x_range = np.linspace(0, np.percentile(envelope, 99), 200)
    ax1.plot(x_range, kde(x_range), 'k-', linewidth=2, alpha=0.8, label='KDE')
    
    ax1.set_xlabel('Envelope Width |p_sup - p_ref|', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title('Distribution of Prediction Disagreement', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.percentile(envelope, 99))
    
    # 2. Cumulative Distribution
    ax2 = plt.subplot(3, 3, (2, 5))
    sorted_env = np.sort(envelope)
    cumulative = np.arange(1, len(sorted_env) + 1) / len(sorted_env) * 100
    ax2.plot(sorted_env, cumulative, linewidth=2.5, color='darkblue')
    
    # Add reference lines
    for pct, label in [(50, '50%'), (90, '90%'), (95, '95%')]:
        val = np.percentile(envelope, pct)
        ax2.axhline(pct, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax2.plot(val, pct, 'ro', markersize=8)
        ax2.text(val, pct + 3, f'{val:.3f}', fontsize=9, ha='center')
    
    ax2.set_xlabel('Envelope Width |p_sup - p_ref|', fontsize=13)
    ax2.set_ylabel('Cumulative Percentage', fontsize=13)
    ax2.set_title('CDF: What % of Predictions Have Envelope < X?', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.percentile(envelope, 99))
    ax2.set_ylim(0, 100)
    
    # 3. Directional Bias Distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(bias, bins=80, alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
    ax3.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax3.axvline(np.mean(bias), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(bias):.3f}')
    ax3.set_xlabel('Directional Bias (p_sup - p_ref)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Who\'s More Optimistic?', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Envelope by Correctness
    ax4 = plt.subplot(3, 3, 6)
    correct_env = envelope[correct == 1]
    incorrect_env = envelope[correct == 0]
    
    parts = ax4.violinplot([correct_env, incorrect_env], positions=[1, 2], 
                           showmeans=True, showmedians=True)
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(['Correct', 'Incorrect'])
    ax4.set_ylabel('Envelope Width', fontsize=11)
    ax4.set_title('Disagreement by Correctness', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    ax4.text(1, np.percentile(correct_env, 95), f'μ={np.mean(correct_env):.3f}',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.text(2, np.percentile(incorrect_env, 95), f'μ={np.mean(incorrect_env):.3f}',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Envelope by Confidence Bins (Separate for p_sup and p_ref)
    ax5 = plt.subplot(3, 3, 7)
    confidence_bins = [(0, 0.3, 'Low'), (0.3, 0.5, 'Mid-Low'), 
                       (0.5, 0.7, 'Mid-High'), (0.7, 1.0, 'High')]
    
    env_by_psup = []
    env_by_pref = []
    labels = []
    
    for low, high, label in confidence_bins:
        # Separate bins for p_sup
        mask_psup = (psup >= low) & (psup < high)
        if np.sum(mask_psup) > 0:
            env_by_psup.append(envelope[mask_psup])
        else:
            env_by_psup.append([])
        
        # Separate bins for p_ref
        mask_pref = (pref >= low) & (pref < high)
        if np.sum(mask_pref) > 0:
            env_by_pref.append(envelope[mask_pref])
        else:
            env_by_pref.append([])
        
        labels.append(label)
    
    # Create positions for grouped box plots
    positions_psup = np.arange(len(labels)) * 2 - 0.3
    positions_pref = np.arange(len(labels)) * 2 + 0.3
    
    bp1 = ax5.boxplot(env_by_psup, positions=positions_psup, widths=0.5, 
                      patch_artist=True, showfliers=False)
    bp2 = ax5.boxplot(env_by_pref, positions=positions_pref, widths=0.5,
                      patch_artist=True, showfliers=False)
    
    # Color the boxes
    for patch in bp1['boxes']:
        patch.set_facecolor('royalblue')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    
    # Set x-axis labels
    ax5.set_xticks(np.arange(len(labels)) * 2)
    ax5.set_xticklabels(labels)
    ax5.set_ylabel('Envelope Width', fontsize=11)
    ax5.set_xlabel('Confidence Level', fontsize=11)
    ax5.set_title('Disagreement by Confidence (Separate)', fontsize=12, fontweight='bold')
    ax5.legend([bp1['boxes'][0], bp2['boxes'][0]], ['p_sup', 'p_ref'], 
               loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Percentile Visualization (Bar Chart)
    ax6 = plt.subplot(3, 3, 8)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(envelope, p) for p in percentiles]
    percentile_labels = [f'{p}th' for p in percentiles]
    
    # Create gradient colors from green (good) to red (bad)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(percentiles)))
    
    bars = ax6.barh(percentile_labels, percentile_values, color=colors, 
                    edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, percentile_values)):
        ax6.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Add vertical reference lines
    ax6.axvline(0.10, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax6.axvline(0.30, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax6.axvline(0.50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    ax6.set_xlabel('Envelope Width', fontsize=11)
    ax6.set_ylabel('Percentile', fontsize=11)
    ax6.set_title('Envelope Percentiles', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.set_xlim(0, max(percentile_values) * 1.15)
    
    # 7. Q-Q Plot (Normality Check)
    ax7 = plt.subplot(3, 3, 9)
    stats.probplot(envelope, dist="norm", plot=ax7)
    ax7.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Envelope Distribution Analysis: Comprehensive View of p_sup vs p_ref Disagreement', 
                 fontsize=17, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    output_path = os.path.join(output_dir, 'envelope_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved envelope distribution to {output_path}")
    
    # Generate detailed statistics
    stats_summary = {
        'envelope_statistics': {
            'count': int(len(envelope)),
            'mean': float(np.mean(envelope)),
            'median': float(np.median(envelope)),
            'std': float(np.std(envelope)),
            'min': float(np.min(envelope)),
            'max': float(np.max(envelope)),
            'percentiles': {
                f'p{p}': float(np.percentile(envelope, p)) 
                for p in [10, 25, 50, 75, 90, 95, 99]
            },
            'iqr': float(np.percentile(envelope, 75) - np.percentile(envelope, 25))
        },
        'directional_bias': {
            'mean': float(np.mean(bias)),
            'median': float(np.median(bias)),
            'psup_higher_pct': float(100 * np.mean(bias > 0)),
            'pref_higher_pct': float(100 * np.mean(bias < 0)),
            'agreement_pct': float(100 * np.mean(bias == 0))
        },
        'by_correctness': {
            'correct_mean_envelope': float(np.mean(envelope[correct == 1])),
            'incorrect_mean_envelope': float(np.mean(envelope[correct == 0])),
            'difference': float(np.mean(envelope[correct == 0]) - np.mean(envelope[correct == 1]))
        },
        'narrow_envelope_cases': {
            'below_0.05': float(100 * np.mean(envelope < 0.05)),
            'below_0.10': float(100 * np.mean(envelope < 0.10)),
            'below_0.15': float(100 * np.mean(envelope < 0.15))
        },
        'wide_envelope_cases': {
            'above_0.30': float(100 * np.mean(envelope > 0.30)),
            'above_0.40': float(100 * np.mean(envelope > 0.40)),
            'above_0.50': float(100 * np.mean(envelope > 0.50))
        }
    }
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'envelope_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    # Print key findings
    print("\n=== Envelope Distribution Summary ===")
    print(f"Total Predictions: {len(envelope):,}")
    print(f"\nCentral Tendency:")
    print(f"  Mean Envelope: {np.mean(envelope):.4f}")
    print(f"  Median Envelope: {np.median(envelope):.4f}")
    print(f"  Std Dev: {np.std(envelope):.4f}")
    print(f"\nDistribution Shape:")
    print(f"  10th Percentile: {np.percentile(envelope, 10):.4f}")
    print(f"  25th Percentile: {np.percentile(envelope, 25):.4f}")
    print(f"  75th Percentile: {np.percentile(envelope, 75):.4f}")
    print(f"  90th Percentile: {np.percentile(envelope, 90):.4f}")
    print(f"  IQR: {np.percentile(envelope, 75) - np.percentile(envelope, 25):.4f}")
    print(f"\nDirectional Bias:")
    print(f"  p_sup > p_ref: {100 * np.mean(bias > 0):.1f}%")
    print(f"  p_ref > p_sup: {100 * np.mean(bias < 0):.1f}%")
    print(f"  Mean Bias: {np.mean(bias):.4f}")
    print(f"\nAgreement Levels:")
    print(f"  Narrow (<0.10): {100 * np.mean(envelope < 0.10):.1f}%")
    print(f"  Moderate (0.10-0.30): {100 * np.mean((envelope >= 0.10) & (envelope < 0.30)):.1f}%")
    print(f"  Wide (>0.30): {100 * np.mean(envelope >= 0.30):.1f}%")
    
    return stats_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="examples/validation/results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model...")
    model, dc, mc, dpath, bkt_params = load_model_from_dir(args.exp_dir, device)
    
    print("Initializing test dataset...")
    model_name = mc.get('model_name', mc.get('model', 'gtransformer'))
    dataset_name = mc.get('dataset_name', mc.get('dataset', 'assist2009'))
    dc['dataset_name'] = dataset_name 
    test_loader, _, _, _ = init_test_datasets(dc, model_name, 64)
    
    print("Collecting envelope data...")
    data = collect_envelope_data(model, test_loader, device)
    
    print(f"Collected {len(data['envelope']):,} predictions")
    
    print("Generating distribution visualizations...")
    stats = plot_envelope_distribution(data, args.output_dir)

if __name__ == "__main__":
    main()
