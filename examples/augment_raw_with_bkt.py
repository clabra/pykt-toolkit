import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

# Add pyBKT to path
try:
    from pyBKT.models import Model
except ImportError:
    print("Error: pyBKT not installed. Run: pip install pyBKT")
    sys.exit(1)

# Import dname2paths from data_preprocess if possible, otherwise define it locally
# (Copying for robustness and to avoid complex relative import issues)
dname2paths = {
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2012": "../data/assist2012/2012-2013-data-with-predictions-4-final.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "../data/nips_task34/train_data/train_task_3_4.csv",
}

# Define column mappings for datasets
# Defaults follow pyBKT expected keys: user_id, skill_name, correct, order_id
dataset_defaults = {
    "assist2009": {
        "order_id": "order_id",
        "skill_name": "skill_name",
        "correct": "correct",
        "user_id": "user_id"
    },
    "assist2015": {
        "order_id": "log_id",
        "skill_name": "sequence_id",
        "correct": "correct",
        "user_id": "user_id"
    },
    "algebra2005": {
        "order_id": "Row",
        "skill_name": "KC(Default)",
        "correct": "Correct First Attempt",
        "user_id": "Anon Student Id",
        "sep": "\t"
    },
    "bridge2algebra2006": {
        "order_id": "Row",
        "skill_name": "KC(SubSkills)",
        "correct": "Correct First Attempt",
        "user_id": "Anon Student Id",
        "sep": "\t"
    },
    "nips_task34": {
        "order_id": "QuestionId",
        "skill_name": "QuestionId",
        "correct": "IsCorrect",
        "user_id": "UserId"
    }
}

def main():
    parser = argparse.ArgumentParser(description='Augment raw datasets with BKT parameters and predictions')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., assist2009)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    
    if args.dataset not in dname2paths:
        print(f"Error: Dataset '{args.dataset}' not found in mapping.")
        sys.exit(1)

    # 1. Setup Directories
    raw_rel_path = dname2paths[args.dataset]
    raw_path = project_root / 'examples' / raw_rel_path # Relative to examples/
    if not raw_path.exists():
        # Try relative to project root
        raw_path = project_root / raw_rel_path.replace('../', '')

    dataset_dir = project_root / 'data' / args.dataset
    bkt_dir = dataset_dir / 'bkt'
    bkt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing dataset: {args.dataset}")
    print(f"Raw file: {raw_path}")
    print(f"BKT directory: {bkt_dir}")

    # 2. Define and Save Defaults (Step 1)
    defaults_path = bkt_dir / 'default_dictionary.json'
    
    if defaults_path.exists():
        print(f"Loading existing column mapping from: {defaults_path}")
        with open(defaults_path, 'r') as f:
            defaults = json.load(f)
    else:
        print(f"No local mapping found. Using registry/generic defaults for '{args.dataset}'")
        defaults = dataset_defaults.get(args.dataset, {
            "order_id": "order_id",
            "skill_name": "skill_name",
            "correct": "correct",
            "user_id": "user_id"
        })
        # Save for future reproducibility
        with open(defaults_path, 'w') as f:
            json.dump(defaults, f, indent=4)
        print(f"Saved defaults to: {defaults_path}")

    # 3. Load Data
    sep = defaults.get('sep', ',')
    try:
        df = pd.read_csv(raw_path, low_memory=False, sep=sep)
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying ISO-8859-1...")
        df = pd.read_csv(raw_path, low_memory=False, encoding='ISO-8859-1', sep=sep)
    
    # Pre-process: handle 'correct' column (-1 to 0 or drop)
    # pyBKT expects 0 or 1.
    original_len = len(df)
    df = df[df[defaults['correct']].isin([0, 1])]
    df = df.dropna(subset=[defaults['skill_name'], defaults['user_id']])
    print(f"Loaded {len(df)} interactions (Dropped {original_len - len(df)} rows with missing or invalid data)")

    # 4. Fit Model (Step 2)
    # Sanitize skill names to avoid regex errors (common in Carnegie Learning datasets)
    print("Sanitizing skill names for BKT fitting...")
    unique_skills = df[defaults['skill_name']].unique()
    skill_to_id = {skill: i for i, skill in enumerate(unique_skills)}
    id_to_skill = {v: k for k, v in skill_to_id.items()}
    
    # Backup original skill names and apply IDs
    original_skills = df[defaults['skill_name']].copy()
    df[defaults['skill_name']] = df[defaults['skill_name']].map(skill_to_id)
    
    model = Model(seed=42, num_fits=1)
    print("Fitting BKT model...")
    # Map column names if they are different from pyBKT defaults in model.fit
    # Note: model.fit 'defaults' argument handles this.
    model.fit(data=df, defaults=defaults)
    
    # Save Model Object (New Requirement)
    model_path = bkt_dir / 'model.pkl'
    model.save(str(model_path))
    print(f"Saved model object to: {model_path}")

    # 5. Extract and Save Parameters (Step 3)
    params_df = model.params()
    skills = params_df.index.get_level_values('skill').unique()
    
    export_params = {}
    for skill_id in skills:
        orig_skill = id_to_skill[int(skill_id)]
        export_params[str(orig_skill)] = {
            'prior': float(params_df.loc[(skill_id, 'prior', 'default'), 'value']),
            'learns': float(params_df.loc[(skill_id, 'learns', 'default'), 'value']),
            'slips': float(params_df.loc[(skill_id, 'slips', 'default'), 'value']),
            'guesses': float(params_df.loc[(skill_id, 'guesses', 'default'), 'value']),
        }
    
    params_path = bkt_dir / 'parameters.json'
    with open(params_path, 'w') as f:
        json.dump(export_params, f, indent=4)
    print(f"Saved parameters to: {params_path}")

    # 6. Augment Dataset (Step 4)
    print("Augmenting dataset with population parameters...")
    # NOTE: We removed model.predict() and 'bkt_p_correct' calculation to avoid 
    # data leakage concerns at the raw level. Row-level predictions will be 
    # handled per-fold in a separate validation cycle.
    
    # Restore original skill names in df for the final CSV
    df[defaults['skill_name']] = original_skills
    
    def get_skill_param(skill, param_name):
        skill_str = str(skill)
        if skill_str in export_params:
            return export_params[skill_str][param_name]
        return np.nan

    df['bkt_p_l0'] = df[defaults['skill_name']].apply(lambda x: get_skill_param(x, 'prior'))
    df['bkt_p_t'] = df[defaults['skill_name']].apply(lambda x: get_skill_param(x, 'learns'))
    df['bkt_p_g'] = df[defaults['skill_name']].apply(lambda x: get_skill_param(x, 'guesses'))
    df['bkt_p_s'] = df[defaults['skill_name']].apply(lambda x: get_skill_param(x, 'slips'))

    # Save augmented file
    output_filename = raw_path.stem + "_bkt.csv"
    output_path = bkt_dir / output_filename
    df.to_csv(output_path, index=False)
    print(f"Saved augmented dataset to: {output_path}")

if __name__ == '__main__':
    main()
