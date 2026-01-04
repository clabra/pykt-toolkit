
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pykt.models import init_model
from pykt.datasets.init_dataset import init_test_datasets
from examples.train_probe import extract_embeddings_and_targets

# --- Config ---
EXP_DIR = "/workspaces/pykt-toolkit/experiments/20251230_224907_idkt_setS-pure_assist2009_baseline_364494"
CHECKPOINT = os.path.join(EXP_DIR, "best_model.pt")
BKT_PREDS = os.path.join(EXP_DIR, "traj_predictions.csv")
OUTPUT_DIR = os.path.join(EXP_DIR, "probing_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selected Students for Trajectories (High, Middle, Low performance)
STUDENT_IDS = [166, 530, 520]
COLORS = ['#e74c3c', '#3498db', '#f1c40f'] # Red, Blue, Yellow

# Set style
sns.set_theme(style="white", context="paper")
plt.rcParams['font.family'] = 'serif'

def plot_trajectories(X_bg, y_bg, student_data, output_path):
    """
    Plots the background population and overlays specific student trajectories.
    student_data is a list of (X_traj, y_traj, uid)
    """
    print("Combining data for t-SNE...")
    all_X = [X_bg]
    for X_t, _, _ in student_data:
        all_X.append(X_t)
    X_combined = np.vstack(all_X)
    
    print(f"Running t-SNE on {len(X_combined)} total points...")
    # PCA pre-reduction
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_combined)
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)
    
    # Split back
    X_bg_tsne = X_tsne[:len(X_bg)]
    curr_idx = len(X_bg)
    
    plt.figure(figsize=(12, 10))
    
    # 1. Plot Background (Population)
    sc = plt.scatter(X_bg_tsne[:, 0], X_bg_tsne[:, 1], c=y_bg, cmap='viridis', alpha=0.1, s=10, label='Population')
    cbar = plt.colorbar(sc)
    cbar.set_label('Reference BKT Mastery', rotation=270, labelpad=15)
    
    # 2. Plot Trajectories
    for i, (X_t, y_t, uid) in enumerate(student_data):
        n_points = len(X_t)
        traj_tsne = X_tsne[curr_idx : curr_idx + n_points]
        curr_idx += n_points
        
        color = COLORS[i % len(COLORS)]
        # Plot points
        plt.scatter(traj_tsne[:, 0], traj_tsne[:, 1], c=color, s=25, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Plot Path with arrows
        for j in range(len(traj_tsne) - 1):
            plt.annotate('', xy=traj_tsne[j+1], xytext=traj_tsne[j],
                         arrowprops=dict(arrowstyle="->", color=color, alpha=0.6, lw=1.5, mutation_scale=15),
                         zorder=4)
        
        # Mark Start and End
        plt.text(traj_tsne[0, 0], traj_tsne[0, 1], f"Start ({uid})", fontsize=10, fontweight='bold', color=color, zorder=6)
        plt.text(traj_tsne[-1, 0], traj_tsne[-1, 1], f"End", fontsize=10, fontweight='bold', color=color, zorder=6)

    plt.title("Individualized Learning Trajectories in iDKT Latent Space", fontsize=16)
    plt.xlabel("t-SNE dimension 1", fontsize=12)
    plt.ylabel("t-SNE dimension 2", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved trajectory plot to {output_path}")

def main():
    # Load config
    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    params = config['defaults']
    dataset_name = config['input']['dataset']
    
    project_root = "/workspaces/pykt-toolkit"
    data_config_path = os.path.join(project_root, 'configs/data_config.json')
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)[dataset_name]
    
    data_config["dataset_name"] = dataset_name
    data_config['dpath'] = os.path.join(project_root, data_config['dpath'].replace("../", ""))
    
    # Init Loader
    test_loader, _, _, _ = init_test_datasets(data_config, 'idkt', params['batch_size'])
    
    # Load Model
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    n_uid = state_dict['student_param.weight'].shape[0] - 1 if 'student_param.weight' in state_dict else 0
    
    model_config = {
        'd_model': params['d_model'], 'd_ff': params['d_ff'], 'num_attn_heads': params['n_heads'],
        'n_blocks': params['n_blocks'], 'dropout': params['dropout'], 'final_fc_dim': params['final_fc_dim'],
        'l2': params['l2'], 'n_uid': n_uid
    }
    model = init_model('idkt', model_config, data_config, params['emb_type'])
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extract All for matching
    bkt_df = pd.read_csv(BKT_PREDS)
    X, y, skills = extract_embeddings_and_targets(model, test_loader, bkt_df, device)
    
    # Organize extracted data by student
    # Since we need to preserve sequence order, we can't just group the raw extracted X, y.
    # We should filter by UID from the results.
    
    # To correctly extract trajectories, we need the raw_uids for each row in X.
    # I'll re-run a simplified extraction to get the mapping.
    # Or I can just use the indices since we know how extract_embeddings_and_targets works.
    
    # Better: Re-run extraction but return a list of (X_uid, y_uid, uid)
    uids_in_results = []
    # (The current extract function in train_probe.py doesn't return uids, 
    # but I can basically replicate it here to be sure of the order).
    
    print("Separating trajectories for selected students...")
    student_data = []
    
    # 1. Background data (sampling)
    bg_mask = ~np.isin(pd.Series(np.zeros(len(X))), STUDENT_IDS) # Placeholder, will fix
    
    # I need to know which row in X belongs to which student.
    # I will modify the extraction logic slightly to identify our students.
    
    # Let's find exactly where student 520, etc. are in the X matrix.
    # Since we don't have UIDs in X, I'll use bkt_df as a reference for the sequence.
    # But extract_embeddings_and_targets only keeps "matched" interactions.
    
    # Simplified approach: Re-extract but keep track of UIDs.
    # I'll just copy-paste the extraction here but modified.
    
    from pykt.utils import set_seed
    set_seed(42)
    
    idx_to_uid = {v: k for k, v in test_loader.dataset.dori['uid_to_index'].items()}
    
    X_bg_list = []
    y_bg_list = []
    
    student_trajs = {uid: [] for uid in STUDENT_IDS}
    student_y = {uid: [] for uid in STUDENT_IDS}
    
    # Re-indexing BKT for faster lookup
    bkt_indexed = {} 
    grouped = bkt_df.groupby('student_id')
    for uid, group in grouped:
        sigs = []
        for _, row in group.iterrows():
            sig = (int(row['skill_id']), int(row['y_true']), round(float(row['p_idkt']), 6))
            sigs.append((sig, float(row['p_bkt'])))
        bkt_indexed[uid] = sigs

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            uids_batch = data["uids"].cpu().numpy().flatten()
            q = data["qseqs"].to(device)
            c = data["cseqs"].to(device)
            r = data["rseqs"].to(device)
            sm = data["smasks"].to(device)
            
            cq_full = torch.cat((q[:, 0:1], data["shft_qseqs"].to(device)), dim=1)
            cc_full = torch.cat((c[:, 0:1], data["shft_cseqs"].to(device)), dim=1)
            cr_full = torch.cat((r[:, 0:1], data["shft_rseqs"].to(device)), dim=1)

            preds, _, _, _, concat_q, _ = model(cc_full.long(), cr_full.long(), pid_data=cq_full.long(), qtest=True, uid_data=data["uids"].to(device))
            
            concat_q_np = concat_q.cpu().numpy()
            preds_np = preds.cpu().numpy()
            mask_np = sm.cpu().numpy()
            cshft_np = data["shft_cseqs"].cpu().numpy()
            rshft_np = data["shft_rseqs"].cpu().numpy()

            for b_idx in range(len(uids_batch)):
                uid = int(idx_to_uid.get(uids_batch[b_idx], uids_batch[b_idx]))
                if uid not in bkt_indexed: continue
                
                student_sigs = bkt_indexed[uid]
                valid_indices = np.where(mask_np[b_idx] == 1)[0]
                
                for t in valid_indices:
                    skill_id = int(cshft_np[b_idx, t])
                    y_true = int(rshft_np[b_idx, t])
                    p_idkt = round(float(preds_np[b_idx, 1+t]), 6)
                    model_sig = (skill_id, y_true, p_idkt)
                    
                    for b_sig, p_bkt in student_sigs:
                        if model_sig == b_sig:
                            emb = concat_q_np[b_idx, 1+t]
                            if uid in STUDENT_IDS:
                                student_trajs[uid].append(emb)
                                student_y[uid].append(p_bkt)
                            else:
                                X_bg_list.append(emb)
                                y_bg_list.append(p_bkt)
                            break
                            
    # Downsample background
    X_bg = np.vstack(X_bg_list)
    y_bg = np.array(y_bg_list)
    if len(X_bg) > 5000:
        idx = np.random.choice(len(X_bg), 5000, replace=False)
        X_bg, y_bg = X_bg[idx], y_bg[idx]
        
    final_student_data = []
    for uid in STUDENT_IDS:
        if student_trajs[uid]:
            final_student_data.append((np.vstack(student_trajs[uid]), np.array(student_y[uid]), uid))
            
    plot_trajectories(X_bg, y_bg, final_student_data, os.path.join(OUTPUT_DIR, "probing_trajectories.png"))

if __name__ == "__main__":
    main()
