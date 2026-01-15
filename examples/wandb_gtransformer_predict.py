import os
import argparse
import json
import copy
import torch
import pandas as pd

# Import the DEDICATED gTransformer evaluation loop
from pykt.models.evaluate_gtransformer import evaluate, evaluate_question
from pykt.models import load_model
from pykt.datasets import init_test_datasets

# CRITICAL ARCHITECTURAL FLAGS
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def main(params):
    if params['use_wandb'] == 1:
        import wandb
        # Load API key if available
        try:
            with open("../configs/wandb.json") as fin:
                wandb_config = json.load(fin)
                os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        except:
            pass
        wandb.init(project="wandb_predict_gtransformer")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path) as fin:
        config = json.load(fin)
    
    # Handle potentially missing model_config in some logging formats
    if "model_config" not in config:
        print(f"[Predict] Config at {save_dir} is missing 'model_config'. Searching for nested config.json...")
        import glob
        nested_configs = glob.glob(os.path.join(save_dir, "**/config.json"), recursive=True)
        nested_configs.sort(key=len, reverse=True)
        found_nested = False
        for nc in nested_configs:
            if os.path.abspath(nc) == os.path.abspath(config_path):
                continue
            try:
                with open(nc) as f:
                    nested_data = json.load(f)
                    if "model_config" in nested_data:
                        config = nested_data
                        found_nested = True
                        break
            except: continue
        
        if not found_nested:
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

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name

    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)

    print(f"Start predicting GTransformer: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")

    ckpt_path = save_dir
    expected_ckpt = os.path.join(save_dir, emb_type + "_model.ckpt")
    
    if not os.path.exists(expected_ckpt):
        import glob
        ckpt_files = glob.glob(os.path.join(save_dir, "**/" + emb_type + "_model.ckpt"), recursive=True)
        if ckpt_files:
            ckpt_files.sort(key=len, reverse=True)
            ckpt_path = os.path.dirname(ckpt_files[0])
            print(f"[Predict] Found checkpoint in subdirectory: {ckpt_path}")

    model = load_model(model_name, model_config, data_config, emb_type, ckpt_path)

    save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")

    # Call the SPECIALIZED evaluation loop
    testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
    window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
  
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        # Call the SPECIALIZED evaluation loop for questions
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
    
    # Persistence
    res_path = os.path.join(save_dir, "eval_results.json")
    with open(res_path, "w") as fout:
        json.dump(dres, fout, indent=4)
    print(f"Results saved to: {res_path}")

    if params['use_wandb'] == 1:
        wandb.log(dres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--fusion_type", type=str, default="late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    main(params)
