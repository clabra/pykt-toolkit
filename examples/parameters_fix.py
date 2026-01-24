#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path

def compute_md5(defaults):
    defaults_json = json.dumps(defaults, sort_keys=True)
    return hashlib.md5(defaults_json.encode()).hexdigest()

def main():
    root_dir = Path(__file__).parent.parent
    param_file = root_dir / "configs" / "parameter_default.json"
    
    with open(param_file, 'r') as f:
        data = json.load(f)
    
    # Add ALL required parameters for all models to defaults if they are missing
    # To satisfy reproducibility audit across the benchmark suite
    new_params = {
        "n_hidden": 128,
        "n_rnn_hidden": 128,
        "n_mlp_hidden": 128,
        "hidden_dim": 64,
        "num_attn_heads": 8,
        "num_en": 4,
        "skill_dim": 256,
        "attention_dim": 64,
        "dim_s": 256,
        "emb_size": 256,
        "n_blocks": 4,
        "d_model": 256
    }
    
    for k, v in new_params.items():
        if k not in data['defaults']:
            data['defaults'][k] = v
            print(f"Added {k}={v} to defaults")

    # Ensure they are in model_config type dictionary
    for k in new_params:
        if k not in data['types']['model_config']:
            data['types']['model_config'][k] = {
                "deprecated": False,
                "ablation": [],
                "description": f"Internal parameter: {k}"
            }

    # Recompute MD5
    data['md5'] = compute_md5(data['defaults'])
    
    with open(param_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated parameter_default.json with all baseline defaults. New MD5: {data['md5']}")

if __name__ == "__main__":
    main()
