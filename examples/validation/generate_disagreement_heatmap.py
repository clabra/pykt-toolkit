
import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic_2d

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from examples.validation.validation_helpers import load_model_from_dir
from pykt.datasets import init_test_datasets

def compute_disagreement_heatmap(model, loader, device):
    """
    Compute envelope disagreement across (P_L0, P_T) parameter space.
    Returns 2D grid of mean envelope widths.
    """
    
    # Storage for all predictions and parameters
    all_p_l0 = []
    all_p_t = []
    all_envelope = []
    all_psup = []
    all_pref = []
    all_correct = []
    
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
            
            # Extract predictions (already shifted)
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
                
                # Calculate envelope width for each prediction
                envelope = np.abs(cur_preds - cur_ref_preds)
                
                # Store all data
                all_p_l0.extend(cur_p_l0)
                all_p_t.extend(cur_p_t)
                all_envelope.extend(envelope)
                all_psup.extend(cur_preds)
                all_pref.extend(cur_ref_preds)
                all_correct.extend(cur_r)
            
            if i > 100:  # Process enough batches for good coverage
                break
    
    # Convert to arrays
    all_p_l0 = np.array(all_p_l0)
    all_p_t = np.array(all_p_t)
    all_envelope = np.array(all_envelope)
    all_psup = np.array(all_psup)
    all_pref = np.array(all_pref)
    all_correct = np.array(all_correct)
    
    print(f"\nCollected {len(all_p_l0):,} predictions")
    print(f"P_L0 range: [{all_p_l0.min():.3f}, {all_p_l0.max():.3f}], mean={all_p_l0.mean():.3f}")
    print(f"P_T range: [{all_p_t.min():.3f}, {all_p_t.max():.3f}], mean={all_p_t.mean():.3f}")
    print(f"Envelope range: [{all_envelope.min():.3f}, {all_envelope.max():.3f}], mean={all_envelope.mean():.3f}")
    
    return {
        'p_l0': all_p_l0,
        'p_t': all_p_t,
        'envelope': all_envelope,
        'psup': all_psup,
        'pref': all_pref,
        'correct': all_correct
    }

def plot_disagreement_heatmap(data, output_dir):
    """
    Generate comprehensive heatmap visualizations.
    """
    
    # Define grid resolution
    n_bins_l0 = 20
    n_bins_t = 20
    
    # Create bins
    l0_bins = np.linspace(0, 1, n_bins_l0 + 1)
    t_bins = np.linspace(0, 1, n_bins_t + 1)
    
    # 1. Mean Envelope Width Heatmap
    env_stat, l0_edges, t_edges, _ = binned_statistic_2d(
        data['p_l0'], data['p_t'], data['envelope'],
        statistic='mean', bins=[l0_bins, t_bins]
    )
    
    # 2. Count heatmap (sample density)
    count_stat, _, _, _ = binned_statistic_2d(
        data['p_l0'], data['p_t'], data['envelope'],
        statistic='count', bins=[l0_bins, t_bins]
    )
    
    # 3. Directional bias (p_sup - p_ref)
    bias = data['psup'] - data['pref']
    bias_stat, _, _, _ = binned_statistic_2d(
        data['p_l0'], data['p_t'], bias,
        statistic='mean', bins=[l0_bins, t_bins]
    )
    
    # 4. Std of envelope (volatility)
    env_std_stat, _, _, _ = binned_statistic_2d(
        data['p_l0'], data['p_t'], data['envelope'],
        statistic='std', bins=[l0_bins, t_bins]
    )
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 14))
    
    # Main heatmap: Mean Envelope Width
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(env_stat.T, origin='lower', aspect='auto', cmap='RdYlGn_r',
                     extent=[0, 1, 0, 1], vmin=0, vmax=0.25)
    ax1.set_xlabel('$P_{L0}$ (Initial Mastery)', fontsize=12)
    ax1.set_ylabel('$P_T$ (Learning Rate)', fontsize=12)
    ax1.set_title('Mean Envelope Width\n(p_sup - p_ref Disagreement)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Mean |p_sup - p_ref|', fontsize=10)
    
    # Add contour lines
    X, Y = np.meshgrid(l0_edges[:-1] + np.diff(l0_edges)/2, 
                       t_edges[:-1] + np.diff(t_edges)/2)
    ax1.contour(X, Y, env_stat.T, levels=5, colors='black', alpha=0.3, linewidths=0.5)
    
    # Sample density
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(np.log10(count_stat.T + 1), origin='lower', aspect='auto', cmap='viridis',
                     extent=[0, 1, 0, 1])
    ax2.set_xlabel('$P_{L0}$ (Initial Mastery)', fontsize=12)
    ax2.set_ylabel('$P_T$ (Learning Rate)', fontsize=12)
    ax2.set_title('Sample Density\n(log10 scale)', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('log10(Count)', fontsize=10)
    
    # Directional bias
    ax3 = plt.subplot(2, 3, 3)
    vmax_bias = max(abs(np.nanmin(bias_stat)), abs(np.nanmax(bias_stat)))
    im3 = ax3.imshow(bias_stat.T, origin='lower', aspect='auto', cmap='RdBu_r',
                     extent=[0, 1, 0, 1], vmin=-vmax_bias, vmax=vmax_bias)
    ax3.set_xlabel('$P_{L0}$ (Initial Mastery)', fontsize=12)
    ax3.set_ylabel('$P_T$ (Learning Rate)', fontsize=12)
    ax3.set_title('Directional Bias\n(p_sup - p_ref)', fontsize=13, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Mean(p_sup - p_ref)', fontsize=10)
    ax3.contour(X, Y, bias_stat.T, levels=[0], colors='black', linewidths=2)
    
    # Envelope volatility (std)
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(env_std_stat.T, origin='lower', aspect='auto', cmap='plasma',
                     extent=[0, 1, 0, 1], vmin=0, vmax=0.15)
    ax4.set_xlabel('$P_{L0}$ (Initial Mastery)', fontsize=12)
    ax4.set_ylabel('$P_T$ (Learning Rate)', fontsize=12)
    ax4.set_title('Disagreement Volatility\n(Std of Envelope Width)', fontsize=13, fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Std(|p_sup - p_ref|)', fontsize=10)
    
    # Marginal distributions
    ax5 = plt.subplot(2, 3, 5)
    # Envelope by P_L0
    l0_marginal = []
    l0_centers = []
    for i in range(n_bins_l0):
        mask = (data['p_l0'] >= l0_bins[i]) & (data['p_l0'] < l0_bins[i+1])
        if np.sum(mask) > 10:
            l0_marginal.append(np.mean(data['envelope'][mask]))
            l0_centers.append((l0_bins[i] + l0_bins[i+1]) / 2)
    ax5.plot(l0_centers, l0_marginal, 'o-', linewidth=2, markersize=6, label='By $P_{L0}$')
    ax5.set_xlabel('$P_{L0}$ (Initial Mastery)', fontsize=12)
    ax5.set_ylabel('Mean Envelope Width', fontsize=12)
    ax5.set_title('Marginal: Disagreement vs Mastery', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    ax6 = plt.subplot(2, 3, 6)
    # Envelope by P_T
    t_marginal = []
    t_centers = []
    for i in range(n_bins_t):
        mask = (data['p_t'] >= t_bins[i]) & (data['p_t'] < t_bins[i+1])
        if np.sum(mask) > 10:
            t_marginal.append(np.mean(data['envelope'][mask]))
            t_centers.append((t_bins[i] + t_bins[i+1]) / 2)
    ax6.plot(t_centers, t_marginal, 's-', linewidth=2, markersize=6, color='coral', label='By $P_T$')
    ax6.set_xlabel('$P_T$ (Learning Rate)', fontsize=12)
    ax6.set_ylabel('Mean Envelope Width', fontsize=12)
    ax6.set_title('Marginal: Disagreement vs Learning Rate', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.suptitle('Prediction Disagreement Analysis: p_sup vs p_ref across BKT Parameter Space', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(output_dir, 'disagreement_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved disagreement heatmap to {output_path}")
    
    # Generate summary statistics
    summary = {
        'overall': {
            'mean_envelope': float(np.mean(data['envelope'])),
            'median_envelope': float(np.median(data['envelope'])),
            'std_envelope': float(np.std(data['envelope'])),
            'p90_envelope': float(np.percentile(data['envelope'], 90)),
            'p95_envelope': float(np.percentile(data['envelope'], 95)),
            'mean_bias': float(np.mean(bias)),
            'psup_higher_pct': float(100 * np.mean(bias > 0)),
        },
        'by_quadrant': {}
    }
    
    # Quadrant analysis
    quadrants = {
        'Low L0 / Low T': (data['p_l0'] < 0.5) & (data['p_t'] < 0.5),
        'Low L0 / High T': (data['p_l0'] < 0.5) & (data['p_t'] >= 0.5),
        'High L0 / Low T': (data['p_l0'] >= 0.5) & (data['p_t'] < 0.5),
        'High L0 / High T': (data['p_l0'] >= 0.5) & (data['p_t'] >= 0.5),
    }
    
    for quad_name, mask in quadrants.items():
        if np.sum(mask) > 0:
            summary['by_quadrant'][quad_name] = {
                'count': int(np.sum(mask)),
                'mean_envelope': float(np.mean(data['envelope'][mask])),
                'mean_bias': float(np.mean(bias[mask])),
                'psup_higher_pct': float(100 * np.mean(bias[mask] > 0)),
            }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'disagreement_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to {summary_path}")
    
    # Print key findings
    print("\n=== Key Findings ===")
    print(f"Overall Mean Envelope Width: {summary['overall']['mean_envelope']:.4f}")
    print(f"Overall Median Envelope Width: {summary['overall']['median_envelope']:.4f}")
    print(f"90th Percentile Envelope: {summary['overall']['p90_envelope']:.4f}")
    print(f"Mean Directional Bias (p_sup - p_ref): {summary['overall']['mean_bias']:.4f}")
    print(f"p_sup Higher: {summary['overall']['psup_higher_pct']:.1f}% of predictions")
    
    print("\n=== By Quadrant ===")
    for quad_name, stats in summary['by_quadrant'].items():
        print(f"{quad_name}:")
        print(f"  Mean Envelope: {stats['mean_envelope']:.4f}")
        print(f"  Mean Bias: {stats['mean_bias']:.4f}")
        print(f"  p_sup Higher: {stats['psup_higher_pct']:.1f}%")
    
    return summary

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
    
    print("Computing disagreement across parameter space...")
    data = compute_disagreement_heatmap(model, test_loader, device)
    
    print("Generating visualizations...")
    summary = plot_disagreement_heatmap(data, args.output_dir)

if __name__ == "__main__":
    main()
