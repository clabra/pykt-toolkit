import json
import os
import glob
import numpy as np

base_pattern = "experiments/20260108_05*_idkt_unconstrained_*"
dirs = glob.glob(base_pattern)

results = []

for d in dirs:
    config_path = os.path.join(d, "config.json")
    results_path = os.path.join(d, "results.json")
    
    if os.path.exists(config_path) and os.path.exists(results_path):
        with open(config_path) as f:
            config = json.load(f)
        with open(results_path) as f:
            res = json.load(f)
            
        dataset = config.get("input", {}).get("dataset", "unknown")
        fold = config.get("input", {}).get("fold", -1)
        auc = res.get("test_auc", 0)
        acc = res.get("test_acc", 0)
        
        results.append({
            "dir": d,
            "dataset": dataset,
            "fold": fold,
            "auc": auc,
            "acc": acc
        })

print(f"Found {len(results)} results.")
df_like = {}
# Group by dataset
for r in results:
    ds = r['dataset']
    if ds not in df_like:
        df_like[ds] = {'aucs': [], 'accs': []}
    df_like[ds]['aucs'].append(r['auc'])
    df_like[ds]['accs'].append(r['acc'])
    print(f"Dataset: {r['dataset']}, Fold: {r['fold']}, AUC: {r['auc']:.4f}, ACC: {r['acc']:.4f}")

print("\n--- Aggregated Results ---")
for ds, metrics in df_like.items():
    mean_auc = np.mean(metrics['aucs'])
    mean_acc = np.mean(metrics['accs'])
    print(f"Dataset: {ds}")
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Mean ACC: {mean_acc:.4f}")
