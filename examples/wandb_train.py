import os
import argparse
import json

import torch
torch.set_num_threads(32) 
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model,evaluate,init_model
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

def main(params):
    if "use_wandb" not in params:
        params['use_wandb'] = 1

    if params['use_wandb']==1:
        import wandb
        # For gainakt2exp, let the specialized training function handle W&B initialization
        if params["model_name"] != "gainakt2exp":
            wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.",fuc_name="main")
    
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "robustkt", "folibikt", "atkt", "lpkt", "skvmn", "dimkt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["simplekt","stablekt", "datakt", "sparsekt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        if model_name in ["dtransformer", "simakt"]:
            train_config["batch_size"] = 1024 ## because of OOM
        model_config = copy.deepcopy(params)
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)
    
    debug_print(text="init_dataset",fuc_name="main")
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level)

    params_str = "_".join([str(v) for k,v in params.items() if not k in ['other_config']])

    print(f"params: {params}, params_str: {params_str}")
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        params_str = params_str+f"_{ str(uuid.uuid4())}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    if model_name in ["dimkt"]:
        # del model_config['num_epochs']
        del model_config['weight_decay']

    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2','batch_size','num_epochs']:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt","stablekt", "datakt","folibikt"]:
        model_config["seq_len"] = seq_len
        
    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")
    
    if model_name == "gainakt2exp":
        # gainakt2exp uses its own model creation, skip init_model
        model = None
        print("gainakt2exp model will be created by train_gainakt2exp_model")
    else:
        # Remove training-only keys that some model constructors (e.g., DTransformer) don't accept
        if 'seq_len' in model_config:
            model_config.pop('seq_len')
        model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        print(f"model is {model}")
    if model_name == "gainakt2exp":
        # gainakt2exp handles optimizer creation internally
        opt = None
    elif model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        print("dtransformer weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "simakt":
        print("simakt weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)
   
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    
    debug_print(text = "train model",fuc_name="main")
    
    if model_name == "gainakt2exp":
        # Use specialized training function for gainakt2exp model
        from train_gainakt2exp import train_gainakt2exp_model
        import argparse
        
        # Convert params to args object that train_gainakt2exp_model expects
        args = argparse.Namespace()
        args.epochs = num_epochs
        args.batch_size = batch_size
        args.lr = learning_rate
        args.weight_decay = params.get('weight_decay', params.get('l2', 1e-5))
        args.patience = params.get('patience', 20)
        args.enhanced_constraints = params.get('enhanced_constraints', True)
        args.monitor_freq = params.get('monitor_freq', 50)
        args.dataset = dataset_name
        args.fold = fold
        args.experiment_suffix = params.get('experiment_suffix', 'wandb')
        args.use_wandb = params.get('use_wandb', 0)
        
        # Pass through individual constraint parameters for ablation studies
        args.non_negative_loss_weight = params.get('non_negative_loss_weight', 0.0)
        args.monotonicity_loss_weight = params.get('monotonicity_loss_weight', 0.05)
        args.mastery_performance_loss_weight = params.get('mastery_performance_loss_weight', 0.5)
        args.gain_performance_loss_weight = params.get('gain_performance_loss_weight', 0.5)
        args.sparsity_loss_weight = params.get('sparsity_loss_weight', 0.1)
        args.consistency_loss_weight = params.get('consistency_loss_weight', 0.3)
        
        # Call the specialized training function
        results = train_gainakt2exp_model(args)
        
        # Extract results in the format expected by wandb_train
        validauc = results['best_val_auc']
        validacc = 0.0  # Not tracked in gainakt2exp training
        best_epoch = len(results['train_history']['val_auc'])  # Last epoch with best performance
        testauc, testacc, window_testauc, window_testacc = -1, -1, -1, -1  # Not computed
        
    elif model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = \
            train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, data_config[dataset_name], fold)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model)
    
    if save_model and model_name != "gainakt2exp":
        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
        best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    
    if model_name == "gainakt2exp":
        model_save_path = f"saved_model/gainakt2exp_{params.get('experiment_suffix', 'wandb')}/best_model.pth"
    else:
        model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
    
    print(f"end:{datetime.datetime.now()}")
    
    if params['use_wandb']==1 and params["model_name"] != "gainakt2exp":
        wandb.log({ 
                    "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch,"model_save_path":model_save_path})
