import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Add pyBKT to path
try:
    from pyBKT.models import Model
except ImportError:
    print("Error: pyBKT not installed. Run: pip install pyBKT")
    sys.exit(1)

def load_bkt_params(model_path):
    """
    Load BKT parameters from a saved pyBKT model.

    Input: model_path (str): Path to the saved pyBKT model.
    Returns: 
        params (dict): Dictionary of parameters per skill. 
        global_params (dict): Dictionary of global parameters.
        model (Model): The loaded pyBKT model. 
    """
    model = Model()
    model.load(model_path)
    params_df = model.params()
    
    params = {}
    skills = params_df.index.get_level_values('skill').unique()
    
    # Calculate global means for fallback
    priors = []
    learns = []
    slips = []
    guesses = []
    
    for skill in skills:
        skill_int = int(skill)
        p = {
            'prior': params_df.loc[(skill, 'prior', 'default'), 'value'],
            'learns': params_df.loc[(skill, 'learns', 'default'), 'value'],
            'slips': params_df.loc[(skill, 'slips', 'default'), 'value'],
            'guesses': params_df.loc[(skill, 'guesses', 'default'), 'value'],
        }
        params[skill_int] = p
        priors.append(p['prior'])
        learns.append(p['learns'])
        slips.append(p['slips'])
        guesses.append(p['guesses'])
    
    global_params = {
        'prior': np.mean(priors),
        'learns': np.mean(learns),
        'slips': np.mean(slips),
        'guesses': np.mean(guesses)
    }
    
    return params, global_params, model

def augment_sequences(df, bkt_params, global_params, model):
    """
    Augment sequences with BKT parameters and mastery trajectories.

    Input: 
        df (pd.DataFrame): DataFrame containing the sequences to be augmented.
        bkt_params (dict): Dictionary of skill parameters.
        global_params (dict): Dictionary of global parameters.
        model (Model): The loaded pyBKT model.
    Returns: 
        df (pd.DataFrame): DataFrame augmented with 2 additional columns for mastery and p(corr) calculated with BKT
    """
    print("Augmenting sequences with BKT data...")
    
    # Prepare lists for new columns
    bkt_p_ls = []   # P(Mastery) trajectory
    bkt_p_corrs = [] # Predicted probability of correctness
    
    # Optional: we could also store the skill-level params if they vary by interaction
    # (though in BKT they are fixed per skill)
    
    for idx, row in df.iterrows():
        concepts = [int(c) for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if r != '-1']
        
        # We need to run forward inference for each student sequence
        # We'll maintain the current state for each skill practiced by this student
        student_mastery = {} # skill -> current P(L)
        
        seq_p_ls = []
        seq_p_corrs = []
        
        for skill, response in zip(concepts, responses):
            # Get params for this skill
            p = bkt_params.get(skill, global_params)
            
            # 1. Initialize mastery if first encounter
            if skill not in student_mastery:
                student_mastery[skill] = p['prior']
            
            p_l = student_mastery[skill]
            
            # 2. Predict probability of correct response (before interaction)
            p_correct = p_l * (1 - p['slips']) + (1 - p_l) * p['guesses']
            
            seq_p_ls.append(f"{p_l:.6f}")
            seq_p_corrs.append(f"{p_correct:.6f}")
            
            # 3. Bayesian update based on observation
            if response == 1:
                p_l_updated = (p_l * (1 - p['slips'])) / max(p_correct, 1e-10)
            else:
                p_l_updated = (p_l * p['slips']) / max(1 - p_correct, 1e-10)
            
            # 4. Learning transition
            p_l_next = p_l_updated + (1 - p_l_updated) * p['learns']
            student_mastery[skill] = np.clip(p_l_next, 0.0, 1.0)
            
        # Pad with -1 to match original length
        pad_len = len(row['concepts'].split(',')) - len(seq_p_ls)
        seq_p_ls.extend(["-1.0"] * pad_len)
        seq_p_corrs.extend(["-1.0"] * pad_len)
        
        bkt_p_ls.append(",".join(seq_p_ls))
        bkt_p_corrs.append(",".join(seq_p_corrs))
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} students...")
            
    df['bkt_mastery'] = bkt_p_ls
    df['bkt_p_correct'] = bkt_p_corrs
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Augment sequences with BKT parameters and mastery')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--bkt_model', type=str, default=None, 
                        help='Path to BKT model .pkl file')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input CSV file (default: processes both train and test sequences)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV file (only used if --input_file is specified)')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    if args.bkt_model is None:
        args.bkt_model = project_root / 'data' / args.dataset / 'bkt_mastery_states.pkl'
        
    print(f"Loading BKT model from: {args.bkt_model}")
    bkt_params, global_params, model = load_bkt_params(args.bkt_model)
    
    files_to_process = []
    if args.input_file:
        output = args.output_file if args.output_file else str(Path(args.input_file).with_suffix('')) + '_bkt.csv'
        files_to_process.append((args.input_file, output))
    else:
        # Default: Process both standard files
        train_file = project_root / 'data' / args.dataset / 'train_valid_sequences.csv'
        test_file = project_root / 'data' / args.dataset / 'test_sequences.csv'
        
        if train_file.exists():
            files_to_process.append((str(train_file), str(train_file.parent / 'train_valid_sequences_bkt.csv')))
        if test_file.exists():
            files_to_process.append((str(test_file), str(test_file.parent / 'test_sequences_bkt.csv')))

    if not files_to_process:
        print("Error: No input files found to process.")
        sys.exit(1)

    for input_path, output_path in files_to_process:
        print(f"Processing: {input_path}")
        df = pd.read_csv(input_path)
        df_augmented = augment_sequences(df, bkt_params, global_params, model)
        print(f"Saving to: {output_path}")
        df_augmented.to_csv(output_path, index=False)
    
    # Save skill-level params separately for the L_param loss
    params_output = project_root / 'data' / args.dataset / 'bkt_skill_params.pkl'
    with open(params_output, 'wb') as f:
        pickle.dump({'params': bkt_params, 'global': global_params}, f)
    print(f"Saved skill-level BKT parameters to: {params_output}")

if __name__ == '__main__':
    main()
