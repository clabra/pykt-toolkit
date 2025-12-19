import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

def gather_results(exp_root="experiments"):
    results = []
    # Find all pareto experiment directories
    dirs = glob.glob(os.path.join(exp_root, "*pareto_l*"))
    
    for d in dirs:
        # Extract lambda from title
        try:
            # Format: ...pareto_l0.5_ID
            parts = os.path.basename(d).split("_")
            l_str = [p for p in parts if p.startswith("l")][0][1:]
            lambda_val = float(l_str)
        except:
            continue
            
        eval_path = os.path.join(d, "eval_results.json")
        interp_path = os.path.join(d, "interpretability_alignment.json")
        
        if os.path.exists(eval_path) and os.path.exists(interp_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            with open(interp_path, 'r') as f:
                interp_data = json.load(f)
                
            results.append({
                'lambda': lambda_val,
                'auc': eval_data.get('test_auc'),
                'corr': interp_data.get('prediction_corr'),
                'mse': interp_data.get('prediction_mse'),
                'im_corr': interp_data.get('initmastery_corr'),
                'rt_corr': interp_data.get('learning_rate_corr'),
                'dir': os.path.basename(d)
            })
            
    if not results:
        print("Gathering results: No completed experiments found yet.")
        print(f"Checked {len(dirs)} directories in {exp_root}.")
        return pd.DataFrame()
        
    print(f"Successfully gathered metrics from {len(results)} experiments.")
    return pd.DataFrame(results).sort_values('lambda')

def plot_pareto(df, output_path="assistant/pareto_frontier.png"):
    if df.empty:
        print("No data to plot.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(df['corr'], df['auc'], 'o-', linewidth=2, markersize=8, color='blue', label='iDKT Frontier')
    
    # Annotate lambdas
    for idx, row in df.iterrows():
        plt.annotate(f"λ={row['lambda']}", (row['corr'], row['auc']), 
                     textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.xlabel('Theory Alignment (Prediction Correlation)')
    plt.ylabel('Predictive Performance (Test AUC)')
    plt.title('iDKT Pareto Frontier: Accuracy vs. Interpretability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    df = gather_results()
    print(df)
    df.to_csv("assistant/pareto_metrics.csv", index=False)
    plot_pareto(df)
