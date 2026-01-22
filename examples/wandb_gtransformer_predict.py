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
    # Merge command-line params (from argparse) into trained_params
    # This allows --dual_eval, --prediction_type, etc. to override config values
    trained_params.update(params)
    # Use merged params for the rest of evaluation
    params = trained_params
    
    fold = params["fold"]
    model_name, dataset_name, emb_type = params["model_name"], params["dataset_name"], params["emb_type"]

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

    # V2.0 compatibility: Merge v2.0 parameters from params into model_config
    # During training, wandb_gtransformer_train.py does model_config = deepcopy(params)
    # During evaluation, we need to ensure v2.0 parameters are in model_config
    v2_params = ['lambda_pca', 'lambda_residual', 'use_population', 'use_traits', 'use_residuals', 'pca_alpha', 'pca_beta']
    for param in v2_params:
        if param in params and param not in model_config:
            model_config[param] = params[param]
    
    model = load_model(model_name, model_config, data_config, emb_type, ckpt_path)

    save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")

    # Dual evaluation mode: run both supervised and reference predictions
    # Uses question-level late fusion protocol (mean averaging across KC predictions per question)
    # CRITICAL: Parameter must come from config (merged with CLI args)
    # No hardcoded defaults - value comes from parameter_default.json or experiment config
    if "dual_eval" not in params:
        raise ValueError(
            "Missing required parameter: dual_eval\\n"
            "The parameter should be in experiment config.json (from parameter_default.json)\\n"
            "or passed explicitly via command line.\\n"
            "This may indicate the experiment was created before dual_eval was added.\\n"
            "For new experiments, ensure parameter_default.json has 'dual_eval' entry."
        )
    dual_eval = params["dual_eval"]
    
    if dual_eval:
        print("\n" + "="*80)
        print("DUAL EVALUATION MODE: Measuring both p_sup and p_ref")
        print("Protocol: Question-level, Late Fusion (Mean Average)")
        print("="*80 + "\n")
        
        # Check if model is grounded (needed for p_ref)
        # v1.0: active_grounding=1 (probing losses)
        # v2.0: use_population=true, use_traits=true, use_residuals=true (three-term decomposition)
        is_grounded_v1 = (hasattr(model, 'active_grounding') and model.active_grounding) or params.get('active_grounding', 0) == 1
        is_grounded_v2 = (hasattr(model, 'use_population') and model.use_population and 
                          hasattr(model, 'use_traits') and model.use_traits)
        is_grounded = is_grounded_v1 or is_grounded_v2
        
        print(f"Model grounding status: v1.0={is_grounded_v1}, v2.0={is_grounded_v2}, final={is_grounded}\n")
        
        # CRITICAL: Dual evaluation must use question-level late fusion protocol
        # This is the ONLY valid way to compare p_sup vs p_ref fairly
        if "test_question_file" not in data_config or test_question_loader is None:
            raise ValueError("Dual evaluation requires test_question_file - cannot use sequence-level metrics")
        
        # Run p_sup evaluation (neural head predictions) - question-level late fusion
        print("\n[1/2] Evaluating p_sup (neural head predictions) - Question-level Late Fusion...")
        sup_save_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions_supervised.txt")
        q_testaucs_sup, q_testaccs_sup = evaluate_question(model, test_question_loader, model_name, 
                                                             fusion_type, sup_save_path, 
                                                             prediction_type="supervised")
        # Use late_mean as primary metric (matches oriauclate_mean naming)
        testauc_sup = q_testaucs_sup.get("late_mean", list(q_testaucs_sup.values())[0])
        testacc_sup = q_testaccs_sup.get("late_mean", list(q_testaccs_sup.values())[0])
        print(f"[supervised] question-level late_mean AUC: {testauc_sup}, ACC: {testacc_sup}")
        
        # Run p_ref evaluation (BKT logic predictions) - only for grounded models
        testauc_ref, testacc_ref = None, None
        q_testaucs_ref, q_testaccs_ref = {}, {}
        if is_grounded:
            print("\n[2/2] Evaluating p_ref (BKT logic predictions) - Question-level Late Fusion...")
            ref_save_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions_reference.txt")
            q_testaucs_ref, q_testaccs_ref = evaluate_question(model, test_question_loader, model_name,
                                                                 fusion_type, ref_save_path,
                                                                 prediction_type="reference")
            testauc_ref = q_testaucs_ref.get("late_mean", list(q_testaucs_ref.values())[0])
            testacc_ref = q_testaccs_ref.get("late_mean", list(q_testaccs_ref.values())[0])
            print(f"[reference] question-level late_mean AUC: {testauc_ref}, ACC: {testacc_ref}")
        else:
            print("\n[2/2] Skipping p_ref evaluation (model not grounded)")
        
        # For backwards compatibility, also run sequence-level evaluation to get extra_metrics
        testauc_seq, testacc_seq, extra_metrics = evaluate(model, test_loader, model_name, 
                                                             save_path=save_test_path, 
                                                             prediction_type="supervised")
        
        # Use question-level metrics as primary
        testauc, testacc = testauc_sup, testacc_sup
    else:
        # Single evaluation mode (legacy behavior)
        prediction_type = params.get("prediction_type", "supervised")
        testauc, testacc, extra_metrics = evaluate(model, test_loader, model_name, save_path=save_test_path, 
                                                    prediction_type=prediction_type)
        print(f"[{prediction_type}] testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    window_testauc_ref, window_testacc_ref = None, None
    
    if dual_eval:
        # Supervised window evaluation
        save_test_window_path_sup = os.path.join(save_dir, model.emb_type+"_test_window_predictions_supervised.txt")
        window_testauc, window_testacc, window_extra_metrics = evaluate(model, test_window_loader, model_name,
                                                                          save_path=save_test_window_path_sup,
                                                                          prediction_type="supervised")
        # Reference window evaluation (if grounded)
        if is_grounded:
            save_test_window_path_ref = os.path.join(save_dir, model.emb_type+"_test_window_predictions_reference.txt")
            window_testauc_ref, window_testacc_ref, _ = evaluate(model, test_window_loader, model_name,
                                                                  save_path=save_test_window_path_ref,
                                                                  prediction_type="reference")
    else:
        save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
        window_testauc, window_testacc, window_extra_metrics = evaluate(model, test_window_loader, model_name, 
                                                                          save_path=save_test_window_path,
                                                                          prediction_type=prediction_type)
    
    print(f"[{params.get('prediction_type', 'supervised')}] testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
  
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
        "prediction_type": params.get("prediction_type", "supervised"),
    }
    
    # Add dual evaluation results if available
    if dual_eval:
        dres["dual_eval"] = True
        dres["protocol"] = "question-level late fusion (mean)"
        
        # Store question-level metrics using standard naming convention
        # oriauclate_mean = p_sup (supervised neural head predictions)
        for key in q_testaucs_sup:
            dres[f"oriauc{key}"] = q_testaucs_sup[key]
        for key in q_testaccs_sup:
            dres[f"oriacc{key}"] = q_testaccs_sup[key]
        
        # Store sequence-level for backwards compatibility
        dres["testauc_sequence_level"] = testauc_seq
        dres["testacc_sequence_level"] = testacc_seq
        
        if testauc_ref is not None:
            # oriauclate_mean_ref = p_ref (BKT logic predictions)
            for key in q_testaucs_ref:
                dres[f"oriauc{key}_ref"] = q_testaucs_ref[key]
            for key in q_testaccs_ref:
                dres[f"oriacc{key}_ref"] = q_testaccs_ref[key]
            
            # Interpretability gap using question-level late_mean metric
            dres["interpretability_gap"] = testauc_sup - testauc_ref
            dres["grounded"] = True
        else:
            dres["grounded"] = False
        
        # Add probing metrics from sequence-level evaluation
        dres.update(extra_metrics)
    else:
        dres["dual_eval"] = False
        # Add extra metrics (probing MSE, etc.)
        dres.update(extra_metrics)
  
    # For non-dual evaluation mode, run question-level evaluation separately
    # (dual_eval already did this as part of the protocol)
    if not dual_eval:
        pred_type = params.get("prediction_type", "supervised")
        if "test_question_file" in data_config and not test_question_loader is None:
            save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
            q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path, prediction_type=pred_type)
            for key in q_testaucs:
                dres["oriauc"+key] = q_testaucs[key]
            for key in q_testaccs:
                dres["oriacc"+key] = q_testaccs[key]
                
        if "test_question_window_file" in data_config and not test_question_window_loader is None:
            save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
            qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path, prediction_type=pred_type)
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
    # NO DEFAULTS - all parameters must be passed explicitly or come from config
    # This enforces reproducibility standards (see examples/reproducibility.md)
    parser.add_argument("--bz", type=int, required=True,
                        help="Batch size for evaluation")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory containing trained model and config.json")
    parser.add_argument("--fusion_type", type=str, required=True,
                        help="Fusion type for multi-skill questions (late_fusion, early_fusion, etc.)")
    parser.add_argument("--use_wandb", type=int, required=True,
                        help="Whether to log to wandb (0 or 1)")
    parser.add_argument("--prediction_type", type=str, required=True,
                        choices=["supervised", "reference"],
                        help="supervised: neural head predictions (p_sup), reference: BKT logic predictions (p_ref)")
    parser.add_argument("--dual_eval", type=int, required=True, choices=[0, 1],
                        help="Run dual evaluation: measure both p_sup and p_ref (0=no, 1=yes)")

    args = parser.parse_args()
    params = vars(args)
    # Convert int to bool for dual_eval
    params['dual_eval'] = bool(params['dual_eval'])
    main(params)
