import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from pyBKT.models import Model
except ImportError:
    print("Error: pyBKT not installed. Run: pip install pyBKT")
    sys.exit(1)

def main():
    dataset = "assist2009"
    raw_path = f"data/{dataset}/skill_builder_data_corrected_collapsed.csv"
    output_path = f"data/{dataset}/skill_builder_data_corrected_collapsed_bkt_augmented.csv"
    
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path, encoding='ISO-8859-1', low_memory=False)
    
    print("Filtering data...")
    df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct", "order_id"])
    
    # We use skill_id as skill_name for pyBKT
    df['skill_name'] = df['skill_id'].astype(str)
    
    # 2. Train/Fit pyBKT
    print("Fitting pyBKT model...")
    model = Model(seed=42, num_fits=1)
    defaults = {'order_id': 'order_id', 'skill_name': 'skill_name', 'correct': 'correct', 'user_id': 'user_id'}
    model.fit(data=df, defaults=defaults)
    
    # 3. Predict metrics
    print("Computing BKT predictions...")
    preds_df = model.predict(data=df)
    
    print(f"Input interactions: {len(df)}")
    print(f"BKT predicted interactions: {len(preds_df)}")
    
    # Robustly merge predictions back to original filtered data
    # pyBKT returns input columns + correct_predictions + state_predictions
    # We'll use order_id and user_id to merge back
    
    # Merge on order_id and user_id as they are unique and stable
    preds_subset = preds_df[['order_id', 'user_id', 'correct_predictions', 'state_predictions']]
    
    # Avoid duplicates in preds_subset if any (though shouldn't be)
    preds_subset = preds_subset.drop_duplicates(subset=['order_id', 'user_id'])
    
    df = pd.merge(df, preds_subset, on=['order_id', 'user_id'], how='left')
    
    # Rename columns to our convention
    df = df.rename(columns={'correct_predictions': 'bkt_p_correct', 'state_predictions': 'bkt_mastery'})
    
    # Handle missing values (for skills pyBKT might have skipped)
    # Use population mean as a fallback
    if df['bkt_p_correct'].isna().any():
        mean_p = df['bkt_p_correct'].mean()
        mean_m = df['bkt_mastery'].mean()
        print(f"Warning: Filling {df['bkt_p_correct'].isna().sum()} missing BKT values with population means.")
        df['bkt_p_correct'] = df['bkt_p_correct'].fillna(mean_p)
        df['bkt_mastery'] = df['bkt_mastery'].fillna(mean_m)

    # 4. Question-Level Fusion (Late Fusion Mean Average)
    print("Applying question-level Late Fusion (Mean Average)...")
    fusion_group = ['user_id', 'order_id']
    df['bkt_p_correct'] = df.groupby(fusion_group)['bkt_p_correct'].transform('mean')
    df['bkt_mastery'] = df.groupby(fusion_group)['bkt_mastery'].transform('mean')
    
    # 5. Save
    print(f"Saving to: {output_path}")
    if 'skill_name' in df.columns:
        df = df.drop(columns=['skill_name'])
        
    df.to_csv(output_path, index=False)
    print("Augmentation complete!")

if __name__ == "__main__":
    main()
