import torch
import numpy as np
import os

from .dkt import DKT
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
from .deep_irt import DeepIRT
from .sakt import SAKT
from .saint import SAINT
from .kqn import KQN
from .atkt import ATKT
from .dkt_forget import DKTForget
from .akt import AKT
from .idkt import iDKT
from .gkt import GKT
from .gkt_utils import get_gkt_graph
from .lpkt import LPKT
from .lpkt_utils import generate_qmatrix
from .skvmn import SKVMN
from .hawkes import HawkesKT
from .iekt import IEKT
from .atdkt import ATDKT
from .simplekt import simpleKT
from .datakt import BAKTTime
from .qdkt import QDKT
from .qikt import QIKT
from .dimkt import DIMKT
from .sparsekt import sparseKT
from .rkt import RKT
from .folibikt import folibiKT
from .dtransformer import DTransformer
from .stablekt import stableKT
from .extrakt import extraKT
from .rekt import ReKT
from .cskt import CSKT
# from .fluckt import FlucKT
from .lefokt_akt import LEFOKT_AKT
from .ukt import UKT
from .hcgkt import HCGKT
from .robustkt import Robustkt
# from .simakt import SimAKT
# from .gainsakt import GainSAKT
from .gainakt2 import GainAKT2
from .gainakt3 import GainAKT3
from .gtransformer import GTransformer
#from .gainakt2_enhanced import GainAKT2Enhanced

device = "cpu" if not torch.cuda.is_available() else "cuda"

def validate_and_apply_ablation_config(model_config, source="config"):
    """
    Ablation Control Center: Validates and applies ablation mode parameter settings.
    
    Behavior:
    - ablation='none': Uses parameters as-is from command/config (no overrides)
    - ablation='all', 'reference', 'probe', 'personalization': Overrides specific parameters
    
    Parameter precedence (for ablation-related params only):
    - ablation mode (from command-line/config) > command-line > config files
    
    Args:
        model_config (dict): Model configuration dictionary (modified in-place)
        source (str): Source of config ("config" or "command_line") for info messages
    
    Returns:
        dict: Validated and updated model_config
    """
    
    # Ablation mode parameter requirements (from Quick Reference Table)
    ABLATION_CONFIGS = {
        "all": {
            "active_grounding": 0,
            "lambda_sup": 1.0,
            "lambda_ref": 0,
            "lambda_probe": 0,
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "description": "Pure Neural (AKT) - all grounding ablated"
        },
        "none": {
            # Special mode: use lambda_* and personalization as-is, but enforce active_grounding
            "active_grounding": 1,
            "description": "No Ablation - use parameters as configured (enforces active_grounding=1)"
        },
        "reference": {
            "active_grounding": 0,
            "lambda_sup": 1.0,
            "lambda_ref": 0,
            "lambda_probe": 0,
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "description": "No Reference Pipeline - BKT grounding ablated"
        },
        "probe": {
            "active_grounding": 1,
            "lambda_probe": 0,
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "description": "No Probe Loss - probe training ablated"
        },
        "personalization": {
            "active_grounding": 1,
            "lambda_initmastery": 0,
            "lambda_rate": 0,
            "personalization": False,
            "description": "No Personalization - student-specific parameters ablated"
        }
    }
    
    # Get ablation mode (default to "none" if not specified)
    ablation = model_config.get("ablation", "none")
    
    # Validate ablation mode
    if ablation not in ABLATION_CONFIGS:
        valid_modes = ", ".join(ABLATION_CONFIGS.keys())
        raise ValueError(
            f"Invalid ablation mode '{ablation}'. "
            f"Valid modes are: {valid_modes}"
        )
    
    # Get configuration for this ablation mode
    ablation_config = ABLATION_CONFIGS[ablation]
    ablation_desc = ablation_config["description"]
    
    # Parameters that can be controlled by ablation
    controlled_params = [
        "active_grounding", "lambda_sup", "lambda_ref", "lambda_probe", 
        "lambda_initmastery", "lambda_rate", "personalization"
    ]
    
    print(f"\n{'='*70}")
    print(f"ABLATION CONTROL CENTER")
    print(f"{'='*70}")
    print(f"Ablation mode: '{ablation}' ({ablation_desc})")
    
    # Special handling for 'none': enforce active_grounding, use others as-is
    if ablation == "none":
        print(f"Using parameters as configured (enforces active_grounding=1):")
        print()
        # Apply active_grounding override
        old_active_grounding = model_config.get('active_grounding', 'not set')
        model_config['active_grounding'] = 1
        
        for param in controlled_params:
            value = model_config.get(param, None)
            if value is None:
                # Skip parameters not in model_config (they may be in train_config)
                continue
            if param == 'active_grounding' and old_active_grounding != 1:
                print(f"  {param:20s} = {value}  (enforced: {old_active_grounding} → 1)")
            else:
                print(f"  {param:20s} = {value}")
        print(f"{'='*70}\n")
        return model_config
    
    # For other ablation modes: apply overrides
    print(f"Applying ablation overrides (ablation config takes precedence):")
    print()
    
    overrides_applied = []
    for param in controlled_params:
        if param in ablation_config:  # Only override if specified in ablation config
            old_value = model_config.get(param, "not set")
            new_value = ablation_config[param]
            model_config[param] = new_value
            
            if old_value != "not set" and old_value != new_value:
                # Type normalization for comparison
                try:
                    if isinstance(new_value, bool):
                        old_normalized = bool(old_value)
                    else:
                        old_normalized = float(old_value)
                        new_value_check = float(new_value)
                    
                    if (isinstance(new_value, bool) and old_normalized != new_value) or \
                       (not isinstance(new_value, bool) and abs(old_normalized - new_value_check) > 1e-9):
                        status = f"OVERRIDE: {old_value} → {new_value}"
                        overrides_applied.append(param)
                    else:
                        status = f"✓ (unchanged: {new_value})"
                except:
                    status = f"OVERRIDE: {old_value} → {new_value}"
                    overrides_applied.append(param)
            else:
                status = "✓ set" if old_value == "not set" else f"✓ (unchanged: {new_value})"
            
            print(f"  {param:20s} = {str(new_value):5}  {status}")
    
    print()
    if overrides_applied:
        print(f"Overridden parameters: {', '.join(overrides_applied)}")
    else:
        print(f"No overrides needed (all parameters already match ablation mode)")
    print(f"{'='*70}\n")
    
    return model_config

def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "deep_irt":
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "idkt":
        model = iDKT(data_config["num_c"], data_config["num_q"], **model_config).to(device)
    elif model_name == "lefokt_akt":
        model = LEFOKT_AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "extrakt":
        model = extraKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "folibikt":
        model = folibiKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"], 
                    data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config,graph=graph,emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)   
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        # print("===before init weights"+"@"*100)
        # model.printparams()
        model.apply(model.init_weights)
        # print("===after init weights")
        # model.printparams()
        model = model.to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)   
    elif model_name == "qdkt":
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qikt":
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "datakt":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rekt":
        model = ReKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type).to(device)
    elif model_name == "stablekt":
        model = stableKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dimkt":
        model = DIMKT(data_config["num_q"],data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rkt":
        model = RKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    elif model_name == "cskt":
        model = CSKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device) 
    # elif model_name == "fluckt":
    #     model = FlucKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "ukt":
        model = UKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "hcgkt":
        model = HCGKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "robustkt":
        model = Robustkt(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dtransformer":
        model = DTransformer(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                     emb_path=data_config["emb_path"]).to(device)      
    # elif model_name == "simakt":
    #     model = SimAKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, 
    #                   emb_path=data_config["emb_path"]).to(device)
    # elif model_name == "gainsakt":
    #     model = GainSAKT(data_config["num_c"], **model_config, emb_type=emb_type, 
    #                     emb_path=data_config["emb_path"]).to(device)
    elif model_name == "gainakt2":
        # Filter out training-specific parameters
        excluded_params = ['learning_rate', 
                          'non_negative_loss_weight', 'consistency_loss_weight', 
                          'use_wandb', 'add_uuid', 'num_epochs']
        gainakt2_config = {k: v for k, v in model_config.items() if k not in excluded_params}
        model = GainAKT2(data_config["num_c"], **gainakt2_config, emb_type=emb_type, 
                        emb_path=data_config["emb_path"]).to(device)
    elif model_name == "gainakt3":
        # Filter out training-specific parameters
        excluded_params = ['learning_rate', 
                          'non_negative_loss_weight', 'consistency_loss_weight', 
                          'use_wandb', 'add_uuid', 'num_epochs']
        gainakt3_config = {k: v for k, v in model_config.items() if k not in excluded_params}
        model = GainAKT3(data_config["num_c"], **gainakt3_config, emb_type=emb_type, 
                        emb_path=data_config["emb_path"]).to(device)
    # elif model_name == "gainakt2_enhanced":
    #     # Filter out training-specific and legacy parameters for enhanced model
    #     excluded_params = ['learning_rate', 'use_gain_head', 'use_mastery_head', 
    #                       'non_negative_loss_weight', 'consistency_loss_weight', 
    #                       'use_wandb', 'add_uuid', 'num_epochs']
    #     enhanced_config = {k: v for k, v in model_config.items() if k not in excluded_params}
    #     model = GainAKT2Enhanced(data_config["num_c"], **enhanced_config, emb_type=emb_type, 
    #                            emb_path=data_config["emb_path"]).to(device)
    elif model_name == "gtransformer":
        # Avoid duplicate keys if they are already in model_config due to reproducibility standards
        _model_config = model_config.copy()
        _model_config.pop("emb_type", None)
        _model_config.pop("emb_path", None)
        
        # Parameter name normalization: handle legacy config files
        # Some configs use num_attn_heads instead of n_heads
        if "n_heads" not in _model_config and "num_attn_heads" in _model_config:
            _model_config["n_heads"] = _model_config.pop("num_attn_heads")
        
        # === ABLATION CONTROL CENTER ===
        # Validate and apply ablation configuration
        try:
            _model_config = validate_and_apply_ablation_config(_model_config, source="init_model")
        except ValueError as e:
            print(str(e))
            import sys
            sys.exit(1)
        # === END ABLATION CONTROL CENTER ===
        
        # Personalization control: enable/disable student-specific embeddings
        # Backward compatibility: if personalization flag is not present, use explicit n_uid if provided
        if "personalization" in _model_config:
            # New behavior: use personalization flag
            if _model_config.get("personalization", False):
                _model_config["n_uid"] = data_config.get("n_uid", 0)
            else:
                _model_config["n_uid"] = 0
        else:
            # Backward compatibility: use explicit n_uid from model_config or default to 0
            if "n_uid" not in _model_config:
                _model_config["n_uid"] = data_config.get("n_uid", 0)
            # If n_uid is already in _model_config, keep it as is
        
        model = GTransformer(data_config["num_c"], data_config["num_q"], **_model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
