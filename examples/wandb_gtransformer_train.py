import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit-compliant GTransformer Training wrapper")
    
    # 1. Launcher-only parameters (Excluded from audit but used for identification)
    parser.add_argument("--model", type=str, required=True, help="Canonical model name")
    parser.add_argument("--dataset", type=str, required=True, help="Canonical dataset name")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--short_title", type=str, required=True)
    
    # 2. Canonical Data & Training Parameters (Mapping to legacy names)
    parser.add_argument("--epochs", type=int, required=True, help="Canonical naming for num_epochs")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--gradient_clip", type=float, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--emb_type", type=str, required=True)
    parser.add_argument("--emb_path", type=str, required=True)
    
    # 3. Model architecture
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True, help="Canonical naming for num_attn_heads")
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--final_fc_dim", type=int, required=True)
    parser.add_argument("--kq_same", type=int, required=True)
    parser.add_argument("--separate_qa", type=int, required=True)
    parser.add_argument("--pretrain_dim", type=int, required=True)
    parser.add_argument("--l2", type=float, required=True)
    parser.add_argument("--l2_rasch", type=float, required=True)
    
    # 4. Ablation and Individualization
    parser.add_argument("--ablation", type=str, required=True)
    parser.add_argument("--n_uid", type=int, required=True)
    
    # 5. Infrastructure
    parser.add_argument("--use_wandb", type=int, required=True)
    parser.add_argument("--add_uuid", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    
    # 6. Benchmark-wide Compatibility Parameters (Ignored by GTransformer but required for Audit)
    parser.add_argument("--lambda_student", type=float, required=True)
    parser.add_argument("--lambda_gap", type=float, required=True)
    parser.add_argument("--lambda_ref", type=float, required=True)
    parser.add_argument("--lambda_initmastery", type=float, required=True)
    parser.add_argument("--lambda_rate", type=float, required=True)
    parser.add_argument("--theory_guided", type=int, required=True)
    parser.add_argument("--calibrate", type=int, required=True)
    parser.add_argument("--grounded_init", type=int, required=True)
    parser.add_argument("--size_m", type=int, required=True)
    parser.add_argument("--graph_type", type=str, required=True)
    parser.add_argument("--answer_dim", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--lambda_r", type=float, required=True)
    parser.add_argument("--lambda_w1", type=float, required=True)
    parser.add_argument("--lambda_w2", type=float, required=True)
    parser.add_argument("--n_hidden", type=int, required=True)
    parser.add_argument("--n_rnn_hidden", type=int, required=True)
    parser.add_argument("--n_mlp_hidden", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--num_attn_heads", type=int, required=True)
    parser.add_argument("--num_en", type=int, required=True)
    parser.add_argument("--skill_dim", type=int, required=True)
    parser.add_argument("--attention_dim", type=int, required=True)
    parser.add_argument("--dim_s", type=int, required=True)
    parser.add_argument("--emb_size", type=int, required=True)
    parser.add_argument("--fusion_type", type=str, required=True)
    parser.add_argument("--bkt_filter", action='store_true') # Boolean flags don't need required=True
    parser.add_argument("--bkt_guess_threshold", type=float, required=True)
    parser.add_argument("--bkt_slip_threshold", type=float, required=True)
    parser.add_argument("--_doc_grounding", type=str, required=True)
    parser.add_argument("--_doc_regularization", type=str, required=True)
    
    # 7. Legacy mapping for model_name, etc.
    parser.add_argument("--model_name", type=str, help="Legacy naming")
    parser.add_argument("--dataset_name", type=str, help="Legacy naming")
    parser.add_argument("--num_epochs", type=int, help="Legacy naming")
    
    args = parser.parse_args()
    params = vars(args)
    
    # Resolve Mappings (Canonical -> Legacy expected by wandb_train.py)
    if 'model' in params: params['model_name'] = params['model']
    if 'dataset' in params: params['dataset_name'] = params['dataset']
    if 'epochs' in params: params['num_epochs'] = params['epochs']
    if 'n_heads' in params: params['num_attn_heads'] = params['n_heads']
    
    main(params)
