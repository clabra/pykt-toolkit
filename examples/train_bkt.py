"""
Compute the mastery state of all the skills for the student at each interaction using pyBKT (Bayesian Knowledge Tracing)

Command: 
python train_bkt.py --dataset assist2015
python train_bkt.py --dataset assist2015 --prepare_data --output_path data/assist2015/bkt_mastery_states.pkl

This script uses pyBKT to:
1. Learn BKT parameters (P_L0, P_T, P_S, P_G) from training data
2. Run forward inference to compute P(learned) at each timestep
3. Save results 

Parameters: 
--dataset: str, required=True, help='Dataset name (e.g., assist2015)'
--output_path: str, default=None, help='Output path for targets (default: data/{dataset}/bkt_mastery_states.pkl, data/{dataset}/bkt_mastery_states_mono.pkl)')

Read data from: 
data_path = f'data/{args.dataset}/train_valid_sequences.csv

Output Files: 
data/[dataset_name]/bkt_mastery_states.pkl

# Train a simple BKT model on one skill in the dataset
# Note that calling fit deletes any previous trained BKT model!
model.fit(data_path, skills = "Plot imperfect radical")

# Train a simple BKT model on multiple skills in the dataset
model.fit(data_path, skills = ["Plot imperfect radical", "Plot pi"])

# Train a multiguess and slip BKT model on multiple skills in the
# dataset. Note: if you are not using Assistments data, you may need 
# to provide a column mapping for the guess/slip classes to use 
# (i.e. if the column name is gsclasses, you would specify 
# multigs = 'gsclasses' or specify a defaults dictionary
# defaults = {'multigs': 'gsclasses'}).
model.fit(data_path, skills = ["Plot imperfect radical", "Plot pi"], multigs = True)

# We can combine multiple model variants.
model.fit(data_path = 'ct.csv', skills = ["Plot imperfect radical", "Plot pi"], multigs = True, forgets = True, multilearn = True)

# We can use a different column to specify the different learn and 
# forget classes. In this case, we use student ID.
model.fit(data_path = 'ct.csv', skills = ["Plot imperfect radical", "Plot pi"], multigs = True, forgets = True, multilearn = 'Anon Student Id')

"""

import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
import sys
import os

# Add pyBKT to path
try:
    from pyBKT.models import Model
    from pyBKT.models import Roster
except ImportError:
    print("Error: pyBKT not installed. Run: pip install pyBKT")
    sys.exit(1)

def prepare_bkt_data(df):
    """
    Convert pykt sequence format to pyBKT format.
    
    Input: dataframe with uid (user_id), concepts (skill_name), responses (correct), selectmasks (order_id)
    Output: pyBKT format with user_id, skill_name, correct, order_id
    """
    records = []
    
    for idx, row in df.iterrows():
        uid = row['uid']
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        selectmasks = [int(m) for m in row['selectmasks'].split(',') if m != '-1']
        
        # Convert to row-per-interaction format
        for order_id, (skill, response, mask) in enumerate(zip(concepts, responses, selectmasks)):
            if mask == 1:
                records.append({
                    'user_id': uid,
                    'skill_name': skill,
                    'correct': response,
                    'order_id': order_id
                })
        
        if (idx + 1) % 1000 == 0:
            print(f"  Prepared {idx + 1}/{len(df)} students...", flush=True)
    
    bkt_df = pd.DataFrame(records)
    return bkt_df

def main():
    parser = argparse.ArgumentParser(description='Compute mastery states using BKT')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., assist2015)')
    parser.add_argument('--output_path', type=str, default=None, 
                       help='Output path for targets (default: data/{dataset}/bkt_mastery_states.pkl)')
    parser.add_argument('--prepare_data', action='store_true', default=False,
                       help='Whether to prepare BKT data (convert from sequences). Default False (load existent data).')
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help='Whether to override existing output file. Default False (load existent data).')
    
    args = parser.parse_args()
    
    # Paths
    # Determine absolute paths based on script location to be robust to CWD
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load data config to get the correct dpath for each dataset
    import json
    config_path = project_root / 'configs' / 'data_config.json'
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    if args.dataset not in data_config:
        raise ValueError(f"Dataset '{args.dataset}' not found in data_config.json")
    
    dataset_config = data_config[args.dataset]
    dpath = dataset_config['dpath']
    
    # Convert relative path to absolute
    if dpath.startswith('../'):
        dpath = str(project_root / dpath[3:])
    
    data_path = Path(dpath) / 'train_valid_sequences.csv'
    
    # Output path: save BKT model to the dataset root (not train_data subdirectory)
    # For nips_task34, this means saving to data/nips_task34/ not data/nips_task34/train_data/
    if args.output_path is None:
        # Extract dataset root from dpath (handle cases like nips_task34/train_data)
        if '/train_data' in str(dpath):
            dataset_root = Path(str(dpath).replace('/train_data', ''))
        else:
            dataset_root = Path(dpath)
        output_path = dataset_root / 'bkt_mastery_states.pkl'
    else:
        output_path = Path(args.output_path)
    
    print(f"\n{'='*80}")
    print("BKT TARGET COMPUTATION")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    
    # Check that input data file exists
    if not data_path.exists():
       raise Exception(f"Input data file {data_path} does not exist")

    # Initialize the model
    model = Model(seed = 42, num_fits = 1)

    # Load data
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} student sequences")

    # Prepare data for pyBKT
    print("\nPreparing data for BKT (converting sequences to interactions)...")
    bkt_df = prepare_bkt_data(df)
    print(f"Prepared {len(bkt_df)} interactions from {bkt_df['user_id'].nunique()} students")

    # Get number of skills
    all_skills = set()
    for concepts_str in df['concepts']:
        skills = [int(c) for c in concepts_str.split(',') if c != '-1']
        all_skills.update(skills)
    num_skills = len(all_skills)
    print(f"Number of unique skills: {num_skills}")
            
    defaults = {
        'order_id': 'order_id', 
        'skill_name': 'skill_name', 
        'correct': 'correct', 
        'user_id': 'user_id'
        }

    # Train BKT model
    print("\nFitting BKT parameters via Expectation-Maximization...")
    model.fit(data=bkt_df, defaults=defaults)
    # Save model
    model.save(output_path)

    print(f"\n{'='*80}")
    print("BKT COMPUTATION COMPLETE")
    print(f"Model saved to: {output_path}")
    print(f"{'='*80}")
    
    # Extract learned parameters
    params_df = model.params()
    params = {}
    skills = params_df.index.get_level_values('skill').unique()
    for skill in skills:
        skill_int = int(skill)
        params[skill_int] = {
            'prior': params_df.loc[(skill, 'prior', 'default'), 'value'],
            'learns': params_df.loc[(skill, 'learns', 'default'), 'value'],
            'slips': params_df.loc[(skill, 'slips', 'default'), 'value'],
            'guesses': params_df.loc[(skill, 'guesses', 'default'), 'value'],
        }

    # Save extracted parameters for easier loading in evaluation scripts
    params_output_path = output_path.parent / 'bkt_skill_params.pkl'
    with open(params_output_path, 'wb') as f:
        pickle.dump({'params': params, 'global': model.coef_}, f)
    print(f"Extracted skill parameters saved to: {params_output_path}")

    # Predict on all skills on the training data.
    # This returns a Pandas DataFrame.
    preds_df = model.predict(data=bkt_df)

    # Evaluate the RMSE of the model on the training data.
    # Note that the default evaluate metric is RMSE.
    training_rmse = model.evaluate(data=bkt_df)

    # Evaluate the AUC of the model on the training data. The supported
    # metrics are AUC, RMSE and accuracy (they should be lowercased in
    # the argument!).
    training_auc = model.evaluate(metric = 'auc', data=bkt_df)

    # We can define a custom metric as well.
    def mae(true_vals, pred_vals):
        """ Calculates the mean absolute error. """
        return np.mean(np.abs(true_vals - pred_vals))

    training_mae = model.evaluate(metric = mae, data=bkt_df)

    # Print evaluatin metrics
    print(f"\nRMSE: {training_rmse}")
    print(f"\nAUC: {training_auc}")
    print(f"\nMAE: {training_mae}")

    # Model file
    print(f"\nBKT file: {output_path}")
    
    
    # Create roster and visualize mastery evolution for a few students
    print(f"\n{'='*80}")
    print("ROSTER INSPECTION: MASTERY TRAJECTORIES")
    print(f"{'='*80}")
    
    unique_students = df['uid'].unique()
    sample_size = min(3, len(unique_students))
    sampled_students = np.random.choice(unique_students, sample_size, replace=False).tolist()
    skills_list = skills.tolist()
    
    # Initialize Roster WITHOUT track_progress (it has a bug in pyBKT)
    # We'll manually track the probabilities as we go
    roster = Roster(students=sampled_students, skills=skills_list, model=model, track_progress=False)
    
    for student_id in sampled_students:
        print(f"\nStudent ID: {student_id}")
        
        # Get historical interactions for this student from bkt_df
        student_interactions = bkt_df[bkt_df['user_id'] == student_id].sort_values('order_id')
        num_interactions = len(student_interactions)
        
        if num_interactions == 0:
            print("  No interactions found for this student.")
            continue
            
        # Get unique skills practiced by this student
        practiced_skills = sorted(student_interactions['skill_name'].unique().astype(int).tolist())
        display_skills = [str(s) for s in practiced_skills[:8]] # Show max 8 skills
        
        # Header for the student's matrix
        print(f"  History of {num_interactions} interactions. Showing mastery state for {len(display_skills)} practiced skills:")
        header_row = f"  {'Step':>4} | {'Skill':>5} | {'Cor':>3} ||" + "".join([f"{' S'+s:>8} |" for s in display_skills])
        print(f"  {'-' * len(header_row)}")
        print(header_row)
        print(f"  {'-' * len(header_row)}")
        
        # Manually track mastery evolution
        mastery_history = {s: [] for s in display_skills}
        
        # Replay interactions and capture state after each update
        for t_idx, (_, row) in enumerate(student_interactions.iterrows()):
            active_skill = str(int(row['skill_name']))
            correctness = int(row['correct'])
            
            # Update roster state
            roster.update_state(active_skill, student_id, correctness)
            
            # Capture current probabilities for all display skills
            probs = []
            for s in display_skills:
                p = roster.get_mastery_prob(s, student_id)
                probs.append(f"{p:8.4f}")
            
            # Print row
            print(f"  {t_idx+1:>4} | {active_skill:>5} | {correctness:>3} ||" + "".join([f"{p} |" for p in probs]))
            
    print(f"\n{'='*80}")
    print("Computed BKT mastery states saved to:", output_path)


if __name__ == '__main__':
    main()
