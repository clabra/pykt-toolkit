import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--emb_type", type=str, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--emb_path", type=str, required=True)
    
    # Model architecture
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_attn_heads", type=int, required=True)
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--final_fc_dim", type=int, required=True)
    parser.add_argument("--kq_same", type=int, required=True)
    parser.add_argument("--separate_qa", type=int, required=True)
    parser.add_argument("--pretrain_dim", type=int, required=True)
    
    # Training parameters
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--gradient_clip", type=float, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--l2", type=float, required=True)
    parser.add_argument("--l2_rasch", type=float, required=True)
    
    # Ablation and Individualization
    parser.add_argument("--ablation", type=str, required=True)
    parser.add_argument("--n_uid", type=int, required=True)
    
    # Infrastructure
    parser.add_argument("--use_wandb", type=int, required=True)
    parser.add_argument("--add_uuid", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    
    # BKT/IDKT Compatibility (passed via model_config but not used by gtransformer yet)
    parser.add_argument("--lambda_student", type=float, required=False, default=1e-5)
    parser.add_argument("--lambda_gap", type=float, required=False, default=1e-5)
    parser.add_argument("--lambda_ref", type=float, required=False, default=0.1)
    parser.add_argument("--lambda_initmastery", type=float, required=False, default=0.1)
    parser.add_argument("--lambda_rate", type=float, required=False, default=0.1)
    parser.add_argument("--theory_guided", type=int, required=False, default=1)
    parser.add_argument("--calibrate", type=int, required=False, default=1)
    parser.add_argument("--size_m", type=int, required=False, default=50)
    parser.add_argument("--graph_type", type=str, required=False, default="transition")
    parser.add_argument("--answer_dim", type=int, required=False, default=96)
    parser.add_argument("--epsilon", type=float, required=False, default=10)
    parser.add_argument("--beta", type=float, required=False, default=0.2)
    parser.add_argument("--lambda_r", type=float, required=False, default=0.01)
    parser.add_argument("--lambda_w1", type=float, required=False, default=0.003)
    parser.add_argument("--lambda_w2", type=float, required=False, default=3.0)
    parser.add_argument("--n_hidden", type=int, required=False, default=128)
    parser.add_argument("--n_rnn_hidden", type=int, required=False, default=128)
    parser.add_argument("--n_mlp_hidden", type=int, required=False, default=128)
    parser.add_argument("--hidden_dim", type=int, required=False, default=64)
    parser.add_argument("--num_en", type=int, required=False, default=4)
    parser.add_argument("--skill_dim", type=int, required=False, default=256)
    parser.add_argument("--attention_dim", type=int, required=False, default=64)
    parser.add_argument("--dim_s", type=int, required=False, default=256)
    parser.add_argument("--emb_size", type=int, required=False, default=256)
    parser.add_argument("--fusion_type", type=str, required=False, default="late_fusion")
    
    args = parser.parse_args()
    params = vars(args)
    main(params)
