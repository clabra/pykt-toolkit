import os
import pandas as pd
import torch
from torch import FloatTensor, LongTensor
from .data_loader import KTDataset

class GTransformerDataset(KTDataset):
    """
    Dataset for GTransformer with support for BKT-augmented columns (Grounded version).
    Loads bkt_p_corrects and bkt_masteries from preprocessed sequences.
    """
    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Modified loader to include bkt_masteries and bkt_p_corrects if present.
        Supports the new 8-line preprocessing format.
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], 
                "smasks": [], "uids": [], "bkt_masteries": [], "bkt_p_corrects": []}

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        
        # Consistent student indexing
        unique_uids = sorted(df["uid"].unique())
        uid_to_index = {uid: idx for idx, uid in enumerate(unique_uids)}
        index_to_uid = {idx: uid for uid, idx in uid_to_index.items()}
        
        interaction_num = 0
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        
        for i, row in df.iterrows():
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
            
            # Augmented columns (Standardizing on plural naming from split_datasets_bkt.py)
            if "bkt_masteries" in row:
                dori["bkt_masteries"].append([float(_) for _ in row["bkt_masteries"].split(",")])
            if "bkt_p_corrects" in row:
                dori["bkt_p_corrects"].append([float(_) for _ in row["bkt_p_corrects"].split(",")])
            
            # Support singular names if they exist (backward compatibility)
            if "bkt_mastery" in row and not dori["bkt_masteries"]:
                dori["bkt_masteries"].append([float(_) for _ in row["bkt_mastery"].split(",")])
            if "bkt_p_correct" in row and not dori["bkt_p_corrects"]:
                dori["bkt_p_corrects"].append([float(_) for _ in row["bkt_p_correct"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            
            dori["uids"].append(uid_to_index[row["uid"]])
            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                if "qidxs" in row: dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                if "rest" in row: dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                if "orirow" in row: dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
                
        # Fallbacks
        if len(dori["qseqs"]) == 0 and len(dori["cseqs"]) > 0:
            dori["qseqs"] = dori["cseqs"]
        if len(dori["tseqs"]) == 0 and len(dori["cseqs"]) > 0:
            dori["tseqs"] = [[0] * len(seq) for seq in dori["cseqs"]]
            
        # Convert to Tensors
        for key in list(dori.keys()):
            if len(dori[key]) == 0: continue
            
            if key == "uids":
                dori[key] = LongTensor(dori[key])
            elif key in ["rseqs", "bkt_masteries", "bkt_p_corrects"]:
                dori[key] = FloatTensor(dori[key])
            else:
                dori[key] = LongTensor(dori[key])

        if "cseqs" in dori:
            mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
            dori["masks"] = mask_seqs
            dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        
        dori["uid_to_index"] = uid_to_index
        dori["index_to_uid"] = index_to_uid
        dori["num_students"] = len(unique_uids)

        if self.qtest:
            for key in list(dqtest.keys()):
                if len(dqtest[key]) > 0:
                    dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return dori, dqtest
        return dori
