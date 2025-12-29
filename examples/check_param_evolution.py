import os
import pandas as pd
import glob
import json

def analyze_params(pattern):
    exp_dirs = glob.glob(pattern)
    exp_dirs.sort()
    
    results = []
    
    for d in exp_dirs:
        config_path = os.path.join(d, "config.json")
        rate_path = os.path.join(d, "traj_rate.csv")
        init_path = os.path.join(d, "traj_initmastery.csv")
        
        if not (os.path.exists(config_path) and os.path.exists(rate_path)):
            continue
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        lambda_val = config.get('overrides', {}).get('lambda_ref', 0.0)
        
        df_rate = pd.read_csv(rate_path)
        df_init = pd.read_csv(init_path)
        
        # Calculate standard deviation of iDKT parameters
        rate_std = df_rate['idkt_rate'].std()
        rate_mean = df_rate['idkt_rate'].mean()
        
        init_std = df_init['idkt_im'].std()
        init_mean = df_init['idkt_im'].mean()
        
        # Also check correlation with BKT
        rate_corr = df_rate['idkt_rate'].corr(df_rate['bkt_rate'])
        init_corr = df_init['idkt_im'].corr(df_init['bkt_im'])
        
        results.append({
            'lambda': lambda_val,
            'rate_mean': rate_mean,
            'rate_std': rate_std,
            'rate_corr_bkt': rate_corr,
            'init_mean': init_mean,
            'init_std': init_std,
            'init_corr_bkt': init_corr,
            'l_student': config.get('overrides', {}).get('lambda_student', config.get('lambda_student')),
            'l_init': config.get('overrides', {}).get('lambda_initmastery', config.get('lambda_initmastery')),
            'l_rate': config.get('overrides', {}).get('lambda_rate', config.get('lambda_rate'))
        })
        
    return pd.DataFrame(results)

print("AS2009 Param Evolution:")
df_2009 = analyze_params("experiments/20251226_181543_idkt_assist2009_hr_sweep_l*")
print(df_2009.to_string(index=False))

print("\nAS2015 Param Evolution:")
df_2015 = analyze_params("experiments/20251226_190425_idkt_assist2015_hr_sweep_l*")
print(df_2015.to_string(index=False))
