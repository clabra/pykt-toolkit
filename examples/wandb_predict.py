import os
import argparse
import json
import copy
import torch
import pandas as pd

from pykt.models import evaluate,evaluate_question,load_model
from pykt.datasets import init_test_datasets

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path) as fin:
        config = json.load(fin)
    
    # If config is "flat" (e.g. from run_repro_experiment.py) and missing sections,
    # try to find a nested config.json created by the training script.
    if "model_config" not in config:
        print(f"[Predict] Config at {save_dir} is missing 'model_config'. Searching for nested config.json...")
        import glob
        from pathlib import Path
        nested_configs = glob.glob(os.path.join(save_dir, "**/config.json"), recursive=True)
        # Sort by depth (longest path first) to prioritize the training script's config
        nested_configs.sort(key=len, reverse=True)
        found_nested = False
        nested_ckpt_path = save_dir
        for nc in nested_configs:
            if os.path.abspath(nc) == os.path.abspath(config_path):
                continue
            try:
                with open(nc) as f:
                    nested_data = json.load(f)
                    if "model_config" in nested_data:
                        print(f"[Predict] Found valid nested config at {nc}")
                        config = nested_data
                        found_nested = True
                        nested_ckpt_path = os.path.dirname(nc)
                        break
            except: continue
        
        if not found_nested:
            print("[Predict] Warning: No nested config with 'model_config' found. Attempting to reconstruct from flat params...")
            # Reconstruct from run_repro_experiment.py structure
            if "train_config" in config:
                config["model_config"] = config["train_config"]
            elif "params" in config:
                config["model_config"] = config["params"]
            
            if "params" in config:
                 config["params"]["model_name"] = config["params"].get("model", config["params"].get("model_name"))
                 config["params"]["dataset_name"] = config["params"].get("dataset", config["params"].get("dataset_name"))
                 config["params"]["emb_type"] = config["params"].get("emb_type", "qid")

    model_config = copy.deepcopy(config["model_config"])
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
        if remove_item in model_config:
            del model_config[remove_item]    
    trained_params = config["params"]
    fold = trained_params["fold"]
    model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
    if model_name in ["saint", "sakt", "atdkt", "simakt"]:
        train_config = config["train_config"]
        seq_len = train_config["seq_len"]
        model_config["seq_len"] = seq_len   

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config.get("data_config", {}).get("num_rgap")
            data_config["num_sgap"] = config.get("data_config", {}).get("num_sgap")
            data_config["num_pcount"] = config.get("data_config", {}).get("num_pcount")
        elif model_name == "lpkt":
            data_config["num_at"] = config.get("data_config", {}).get("num_at")
            data_config["num_it"] = config.get("data_config", {}).get("num_it")

    if model_name not in ["dimkt"]:        
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    # Re-verify dynamic fields if they were missing (common in re-sanitized or old configs)
    if model_name in ["dkt_forget", "bakt_time"] and data_config.get("num_rgap") is None:
        print("[Predict] Warning: num_rgap missing in config, inferring from test dataset...")
        data_config["num_rgap"] = test_loader.dataset.max_rgap + 1
        data_config["num_sgap"] = test_loader.dataset.max_sgap + 1
        data_config["num_pcount"] = test_loader.dataset.max_pcount + 1
    elif model_name == "lpkt" and data_config.get("num_at") is None:
         # For LPKT we'd need to re-generate time2idx, but it's complex here.
         # Usually it's present or we fail.
         pass

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    # Determine checkpoint directory
    ckpt_path = save_dir
    expected_ckpt = os.path.join(save_dir, emb_type + "_model.ckpt")
    
    if not os.path.exists(expected_ckpt):
        print(f"[Predict] Checkpoint not found at {expected_ckpt}. Searching subdirectories...")
        import glob
        ckpt_files = glob.glob(os.path.join(save_dir, "**/" + emb_type + "_model.ckpt"), recursive=True)
        if ckpt_files:
            # Pick the one that matches best or is deepest (usually hyperparameter dir)
            ckpt_files.sort(key=len, reverse=True)
            ckpt_path = os.path.dirname(ckpt_files[0])
            print(f"[Predict] Found checkpoint in subdirectory: {ckpt_path}")
        else:
            print(f"[Predict] Warning: No {emb_type}_model.ckpt found in {save_dir} or its subdirectories.")

    model = load_model(model_name, model_config, data_config, emb_type, ckpt_path)

    save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))                

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path)
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
    if model.model_name == "rkt":
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_test_window_path)
    else:
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    # question_testauc, question_testacc = -1, -1
    # question_window_testauc, question_window_testacc = -1, -1
  
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    q_testaucs, q_testaccs = -1,-1
    qw_testaucs, qw_testaccs = -1,-1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
            
    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]

        
    # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    # print(f"question_testauc: {question_testauc}, question_testacc: {question_testacc}, question_window_testauc: {question_window_testauc}, question_window_testacc: {question_window_testacc}")
    
    print(dres)
    config_file = os.path.join(save_dir, "config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            raw_config = json.load(f)
            # Standard PyKT uses 'params', run_repro_experiment uses 'input'
            if 'params' in raw_config:
                dres.update(raw_config['params'])
            elif 'input' in raw_config:
                dres.update(raw_config['input'])
    
    # Structured Results Export: Save to eval_results.json for non-wandb aggregation
    res_path = os.path.join(save_dir, "eval_results.json")
    with open(res_path, "w") as fout:
        json.dump(dres, fout, indent=4)
    print(f"Results saved to: {res_path}")

    if params['use_wandb'] == 1:
        wandb.log(dres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
