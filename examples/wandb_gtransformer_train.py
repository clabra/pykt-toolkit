import os
import argparse
import json
import torch
torch.set_num_threads(32) 
from torch.optim import Adam
import copy
import datetime
import pickle

# Import the DEDICATED gTransformer training loop
from pykt.models.train_gtransformer import train_model
from pykt.models import init_model
from pykt.utils import debug_print, set_seed
from pykt.datasets import init_dataset4train
from torch.utils.data import DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    """Save configuration with explicit documentation for reproducibility."""
    import sys
    
    # Build explicit command string from current invocation
    train_command = " ".join(sys.argv)
    
    d = {
        "train_config": train_config, 
        'model_config': model_config, 
        "data_config": data_config, 
        "params": params,
        "commands": {
            "train_explicit": train_command,
            "launched_at": datetime.datetime.now().isoformat()
        },
        "_documentation": {
            "purpose": "Configuration checkpoint for model training",
            "train_command_info": "The train_explicit command shows the exact command used to launch this training run",
            "note": "For evaluation, use the config.json in the parent experiment folder which contains the complete eval_explicit command"
        }
    }
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout, indent=2)

def main(params):
    if "use_wandb" not in params:
        params['use_wandb'] = 1

    if params['use_wandb']==1:
        import wandb
        wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.", fuc_name="main")
    
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        # Standardize gtransformer batch size
        train_config["batch_size"] = 64
        
        model_config = copy.deepcopy(params)
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            if key in model_config:
                del model_config[key]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']

    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)
    
    debug_print(text="init_dataset", fuc_name="main")
    
    # Phase 3: Active Grounding (Custom Data Loading)
    if params.get('active_grounding', 1) == 1:
        from pykt.datasets.gtransformer_dataloader import GTransformerDataset
        dpath = data_config[dataset_name]["dpath"]
        if dpath.startswith("../"):
            dpath = os.path.join(os.getcwd(), dpath)
        target_path = os.path.join(dpath, "bkt_targets_train_valid.npz")
        
        train_file = os.path.join(dpath, data_config[dataset_name]["train_valid_file"])
        all_folds = set(data_config[dataset_name]["folds"])
        
        curvalid = GTransformerDataset(train_file, data_config[dataset_name]["input_type"], {fold}, target_path=target_path)
        curtrain = GTransformerDataset(train_file, data_config[dataset_name]["input_type"], all_folds - {fold}, target_path=target_path)
        
        nw = int(os.getenv('PYKT_NUM_WORKERS', '32'))
        train_loader = DataLoader(curtrain, batch_size=batch_size, num_workers=nw, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(curvalid, batch_size=batch_size, num_workers=nw, pin_memory=True, shuffle=False)
        print(f"  [Active Grounding] Using GTransformerDataset with targets from {target_path}")
    else:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)

    # Create a compact identifying string for the checkpoint directory
    # Only include essential parameters to avoid "File name too long" errors
    # The full configuration is already saved in config.json
    whitelist = ['model_name', 'dataset_name', 'fold', 'seed', 'd_model', 'n_heads', 'n_blocks', 'learning_rate']
    params_str = "_".join([str(params.get(k, 'none')) for k in whitelist])

    print(f"params_str: {params_str}")
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        params_str = params_str + f"_{str(uuid.uuid4())}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    
    # Dump all parameters that will be used for training
    print("\n" + "="*70)
    print("TRAINING PARAMETERS DUMP (Final Configuration)")
    print("="*70)
    print("\nmodel_config (architecture parameters):")
    for key in sorted(model_config.keys()):
        print(f"  {key:25s} = {model_config[key]}")
    print("\ntrain_config (training hyperparameters):")
    for key in sorted(train_config.keys()):
        print(f"  {key:25s} = {train_config[key]}")
    print("\ndata_config (dataset parameters):")
    for key in sorted(data_config[dataset_name].keys()):
        print(f"  {key:25s} = {data_config[dataset_name][key]}")
    print("="*70 + "\n")

    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2','batch_size','num_epochs']:
        if remove_item in model_config:
            del model_config[remove_item]
    
    # === ABLATION CONTROL CENTER ===
    # Note: The ablation control center is also called in init_model,
    # but we call it here first to validate before saving config
    from pykt.models.init_model import validate_and_apply_ablation_config
    try:
        model_config = validate_and_apply_ablation_config(model_config, source="command_line")
    except ValueError as e:
        print(str(e))
        import sys
        sys.exit(1)
    # === END ABLATION CONTROL CENTER ===
    
    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
        
    debug_print(text="init_model", fuc_name="main")
    print(f"model_name: {model_name}")
    
    # Initialize GTransformer
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model is {model}")
    
    # Theory-Guided Initialization (Crucial for GTransformer)
    if model.ablation != "all":
        dpath = data_config[dataset_name]["dpath"]
        if dpath.startswith("../"):
            dpath = os.path.join(os.getcwd(), dpath)
        
        bkt_path = os.path.join(dpath, "bkt_skill_params.pkl")
        if not os.path.exists(bkt_path):
            bkt_path = os.path.join(dpath, "old", "bkt_skill_params.pkl")
            
        if os.path.exists(bkt_path):
            print(f"  [GTransformer] Loading BKT skill parameters from {bkt_path}")
            with open(bkt_path, "rb") as f:
                bkt_params = pickle.load(f)
            model.load_theory_params(bkt_params)
        else:
            print(f"  [GTransformer] WARNING: No BKT skill parameters found at {bkt_path}")

    # Optimizer
    if optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)
    else:
        # Default fallback
        opt = Adam(model.parameters(), learning_rate)
   
    save_model = True
    
    debug_print(text="train model (Using Specialized GTransformer Loop)", fuc_name="main")
    
    # Call the SPECIALIZED training loop
    testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model)
    
    if save_model:
        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
        best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    
    model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
    
    print(f"end:{datetime.datetime.now()}")
    
    if params['use_wandb'] == 1:
        wandb.log({"validauc": validauc, "validacc": validacc, "best_epoch": best_epoch, "model_save_path": model_save_path})

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
    parser.add_argument("--dual_eval", action='store_true', help="Whether to run dual evaluation")
    parser.add_argument("--personalization", action='store_true', help="Whether to use student personalization")
    parser.add_argument("--prediction_type", type=str, required=True, help="Type of prediction (supervised/reference)")
    
    # 6. Benchmark-wide Compatibility Parameters (Ignored by GTransformer but required for Audit)
    parser.add_argument("--lambda_student", type=float, required=True)
    parser.add_argument("--lambda_gap", type=float, required=True)
    parser.add_argument("--lambda_sup", type=float, required=True, help="Weight for supervised loss on predictions")
    parser.add_argument("--lambda_ref", type=float, required=True, help="Weight for BKT reference supervised loss")
    parser.add_argument("--lambda_initmastery", type=float, required=True, help="Weight for L0 grounding loss")
    parser.add_argument("--lambda_rate", type=float, required=True, help="Weight for T grounding loss")
    parser.add_argument("--active_grounding", type=int, required=True, help="Whether to use active grounding (probing loss)")
    parser.add_argument("--lambda_probe", type=float, required=True, help="Weight for probing loss")
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
