
import torch
import os

checkpoint_path = 'experiments/20251223_193204_idkt_assist2009_baseline_742098/best_model.pt'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if 'student_param.weight' in state_dict:
        vs = state_dict['student_param.weight']
        kc = state_dict['student_gap_param.weight']
        
        print("--- Student Parameter Stats ---")
        print(f"vs (velocity) - mean: {vs.mean().item():.6f}, std: {vs.std().item():.6f}, min: {vs.min().item():.6f}, max: {vs.max().item():.6f}")
        print(f"kc (knowledge) - mean: {kc.mean().item():.6f}, std: {kc.std().item():.6f}, min: {kc.min().item():.6f}, max: {kc.max().item():.6f}")
        
    else:
        print("Student parameters not found in checkpoint.")
else:
    print(f"Checkpoint {checkpoint_path} not found.")
