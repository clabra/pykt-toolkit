import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

def gather_results(exp_root="experiments", prefix="pareto_highres_l"):
    raw_results = []
    # Find all pareto experiment directories
    dirs = glob.glob(os.path.join(exp_root, f"*{prefix}*"))
    
    for d in dirs:
        # Extract lambda from title
        try:
            # Format: YYYYMMDD_HHMMSS_idkt_pareto_highres_l0.55_ID
            basename = os.path.basename(d)
            parts = basename.split("_")
            l_part = [p for p in parts if p.startswith("l") and '.' in p][0]
            lambda_val = float(l_part[1:])
            timestamp = "_".join(parts[0:2])
        except Exception as e:
            continue
            
        eval_path = os.path.join(d, "eval_results.json")
        interp_path = os.path.join(d, "interpretability_alignment.json")
        
        if os.path.exists(eval_path) and os.path.exists(interp_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            with open(interp_path, 'r') as f:
                interp_data = json.load(f)
                
            raw_results.append({
                'lambda': lambda_val,
                'timestamp': timestamp,
                'auc': eval_data.get('test_auc'),
                'corr': interp_data.get('prediction_corr'),
                'mse': interp_data.get('prediction_mse'),
                'im_corr': interp_data.get('initmastery_corr'),
                'rt_corr': interp_data.get('learning_rate_corr'),
                'dir': basename
            })
            
    if not raw_results:
        print("Gathering results: No completed experiments found yet.")
        print(f"Checked {len(dirs)} directories in {exp_root}.")
        return pd.DataFrame()
        
    # Group by lambda and pick the latest timestamp
    df_raw = pd.DataFrame(raw_results)
    # Sort by lambda then by timestamp descending
    df_raw = df_raw.sort_values(['lambda', 'timestamp'], ascending=[True, False])
    # Drop duplicates for each lambda, keeping the first (latest timestamp)
    df_final = df_raw.drop_duplicates('lambda', keep='first')
    
    print(f"Successfully gathered metrics from {len(df_final)} unique lambda experiments.")
    return df_final.sort_values('lambda')

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="pareto_highres_l")
    parser.add_argument("--output", type=str, default="assistant/pareto_metrics_highres.csv")
    parser.add_argument("--plot", type=str, default="assistant/pareto_frontier_highres.png")
    args = parser.parse_args()

    df = gather_results(prefix=args.prefix)
    if not df.empty:
        print(df)
        df.to_csv(args.output, index=False)
        plot_pareto(df, output_path=args.plot)
