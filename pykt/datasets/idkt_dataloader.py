import os
import pandas as pd
import torch
from torch import FloatTensor, LongTensor
from .data_loader import KTDataset

class IDKTDataset(KTDataset):
    """
    Dataset for iDKT with support for BKT-augmented columns.
    """
    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Modified loader to include bkt_mastery and bkt_p_correct if present.
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], 
                "smasks": [], "uids": [], "bkt_mastery": [], "bkt_p_correct": []}

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        
        # Create a deterministic UID-to-index mapping
        # Sort to ensure consistent ordering across runs and folds
        unique_uids = sorted(df["uid"].unique())
        uid_to_index = {uid: idx for idx, uid in enumerate(unique_uids)}
        
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
            
            # Augmented columns
            if "bkt_mastery" in row:
                dori["bkt_mastery"].append([float(_) for _ in row["bkt_mastery"].split(",")])
            if "bkt_p_correct" in row:
                dori["bkt_p_correct"].append([float(_) for _ in row["bkt_p_correct"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            dori["uids"].append(uid_to_index[row["uid"]])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
                
        # 1. Ensure qseqs and tseqs are not empty (standard pykt fallbacks)
        if len(dori["qseqs"]) == 0 and len(dori["cseqs"]) > 0:
            dori["qseqs"] = dori["cseqs"]
        if len(dori["tseqs"]) == 0 and len(dori["cseqs"]) > 0:
            # Create dummy timestamps (all zeros) for models that expect them but don't use them
            dori["tseqs"] = [[0] * len(seq) for seq in dori["cseqs"]]
            
        # 2. Consolidate keys - convert to tensors
        for key in list(dori.keys()):
            if len(dori[key]) == 0:
                # Keep empty list to prevent KeyError, but some scripts might fail if they expect a Tensor
                # Let's convert to an empty Tensor if possible, or just leave as is if not critical
                continue
            
            if key == "uids":
                dori[key] = LongTensor(dori[key])
            elif key in ["rseqs", "bkt_mastery", "bkt_p_correct", "bkt_im", "bkt_p"]:
                dori[key] = FloatTensor(dori[key])
            else:
                dori[key] = LongTensor(dori[key])

        # 3. Create masks and shifted sequences
        # Note: KTDataset usually handles the shifting in __getitem__, 
        # but some pykt loaders also pre-calculate 'masks' and 'smasks' in __load_data__.
        
        if "cseqs" in dori:
            mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
            dori["masks"] = mask_seqs
            dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        
        dori["uid_to_index"] = uid_to_index
        dori["index_to_uid"] = {idx: uid for uid, idx in uid_to_index.items()}
        dori["num_students"] = len(unique_uids)

        if self.qtest:
            for key in list(dqtest.keys()):
                if len(dqtest[key]) > 0:
                    dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return dori, dqtest
        return dori
