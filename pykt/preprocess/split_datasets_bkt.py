import os
import sys
import pandas as pd
import numpy as np
import json
import copy
from .split_datasets import ALL_KEYS, ONE_KEYS, extend_multi_concepts, save_dcur
from .split_datasets import train_test_split, KFold_split, calStatistics, get_max_concepts, id_mapping, write_config, get_inter_qidx, generate_question_sequences

# Updated keys for BKT augmented data
BKT_KEYS = ["bkt_p_corrects", "bkt_masteries"]
ALL_KEYS_BKT = ALL_KEYS + BKT_KEYS

def read_data_bkt(fname, min_seq_len=3, response_set=[0, 1]):
    """Read BKT augmented 8-line format"""
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()
        while i < len(lines):
            line = lines[i].strip()
            # 8 lines per user
            mod = i % 8
            if mod == 0:  # 0. stuid,seqlen
                effective_keys.add("uid")
                tmps = line.split(",")
                stuid, seq_len = tmps[0], int(tmps[1])
                if seq_len < min_seq_len:
                    i += 8
                    dcur = dict()
                    delstu += 1
                    delnum += seq_len
                    continue
                dcur["uid"] = stuid
                goodnum += seq_len
            elif mod == 1:  # 1. questions
                dcur["questions"] = line.split(",") if line != "NA" else []
                if "questions" in dcur: effective_keys.add("questions")
            elif mod == 2:  # 2. concepts
                dcur["concepts"] = line.split(",") if line != "NA" else []
                if "concepts" in dcur: effective_keys.add("concepts")
            elif mod == 3:  # 3. responses
                effective_keys.add("responses")
                dcur["responses"] = [int(r) for r in line.split(",")]
            elif mod == 4:  # 4. timestamps
                dcur["timestamps"] = line.split(",") if line != "NA" else []
                if dcur["timestamps"]: effective_keys.add("timestamps")
            elif mod == 5:  # 5. usetimes
                dcur["usetimes"] = line.split(",") if line != "NA" else []
                if dcur["usetimes"]: effective_keys.add("usetimes")
            elif mod == 6:  # 6. bkt_p_corrects (New)
                dcur["bkt_p_corrects"] = line.split(",") if line != "NA" else []
                if dcur["bkt_p_corrects"]: effective_keys.add("bkt_p_corrects")
            elif mod == 7:  # 7. bkt_masteries (New)
                dcur["bkt_masteries"] = line.split(",") if line != "NA" else []
                if dcur["bkt_masteries"]: effective_keys.add("bkt_masteries")

                # End of user block
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1
            
    df = pd.DataFrame(dres)
    print(f"BKT Loader: Deleted {delstu} students, {delnum} interactions. Good: {goodnum}")
    return df, effective_keys

def generate_sequences_bkt(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    """Wrapper that ensures BKT keys are saved"""
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            j += maxlen
            
        if rest < min_seq_len:
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                # Handle numeric vs string padding
                if key in ["responses", "selectmasks"]:
                    pad_vals = [str(pad_val)] * pad_dim
                else:
                    pad_vals = [str(pad_val)] * pad_dim
                
                paded_info = np.concatenate([dcur[key][j:], pad_vals])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    final_keys = ALL_KEYS_BKT + ["selectmasks"]
    dfinal = dict()
    for key in final_keys:
        if key in dres:
            dfinal[key] = dres[key]
    return pd.DataFrame(dfinal)

def main(dname, fname, dataset_name, configf, min_seq_len=3, maxlen=200, kfold=5):
    """Main split function for BKT augmented data"""
    stares = []
    
    # 1. Read 8-line data
    total_df, effective_keys = read_data_bkt(fname, min_seq_len=min_seq_len)
    
    # Calculate max_concepts for config BEFORE extending multi-concepts
    if 'concepts' in effective_keys:
        from .split_datasets import get_max_concepts
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    # 2. Extend multi concepts (Standard PyKT handles this if '_' in concepts)
    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    
    # 3. ID mapping
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts
    
    save_id2idx = lambda d, p: open(p, "w").write(json.dumps(d))
    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")
    
    # 4. Train/Test Split
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)
    
    # Order keys for saving
    df_save_keys = [k for k in ALL_KEYS_BKT if k in effective_keys]
    
    # 5. Save Splits & Sequences
    splitdf[df_save_keys].to_csv(os.path.join(dname, "train_valid_bkt.csv"), index=None)
    
    split_seqs = generate_sequences_bkt(splitdf, effective_keys, min_seq_len, maxlen)
    split_seqs.to_csv(os.path.join(dname, "train_valid_sequences_bkt.csv"), index=None)
    
    # Test set
    test_df["fold"] = [-1] * test_df.shape[0]
    test_seqs = generate_sequences_bkt(test_df, effective_keys, min_seq_len, maxlen)
    test_seqs.to_csv(os.path.join(dname, "test_sequences_bkt.csv"), index=None)
    test_df[df_save_keys].to_csv(os.path.join(dname, "test_bkt.csv"), index=None)
    
    # Config write with custom filenames
    other_config = {
        "train_valid_original_file": "train_valid_bkt.csv", 
        "train_valid_file": "train_valid_sequences_bkt.csv",
        "test_file": "test_sequences_bkt.csv",
        "test_original_file": "test_bkt.csv",
        "bkt_augmented": True
    }
    
    write_config(dataset_name=dataset_name + "_bkt", dkeyid2idx=dkeyid2idx, effective_keys=effective_keys, 
                configf=configf, dpath=dname, k=kfold, min_seq_len=min_seq_len, maxlen=maxlen, other_config=other_config)
    
    print(f"BKT Augmented Preprocessing Complete: {dname}")
