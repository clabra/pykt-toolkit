
import os
import pandas as pd
import torch
from torch import FloatTensor, LongTensor
import numpy as np
from .data_loader import KTDataset

class GTransformerDataset(KTDataset):
    """
    Dataset for GTransformer with support for BKT soft labels (Active Grounding).
    """
    def __init__(self, file_path, input_type, folds, qtest=False, target_path=None):
        self.target_path = target_path
        self.file_path = file_path
        self.folds = folds
        super().__init__(file_path, input_type, folds, qtest)
        
        # KEY FIX: If we loaded from a pickle, we need to ensure targets are aligned.
        if "target_l0" not in self.dori:
            self._align_targets()

    def _align_targets(self):
        """
        Calculates or loads the targets and attaches them to dori.
        """
        # We need to know which targets belong to which sequences.
        # If we don't have 'ori_row' in dori (it wasn't in the pickle), 
        # we have a problem. But we can reconstruct it from the CSV.
        
        # Safe way: Load the targets and align using the row indices 
        # (assuming KTDataset didn't shuffle or skip rows during loading, 
        # which it doesn't - it loads in CSV order).
        
        if self.target_path and os.path.exists(self.target_path):
            print(f"[GTransformerDataset] Loading BKT targets from {self.target_path}...")
            targets_data = np.load(self.target_path)
            all_target_l0 = FloatTensor(targets_data['target_l0'])
            all_target_t = FloatTensor(targets_data['target_t'])
            
            # Alignment: KTDataset loads folds in order.
            # We need the original row indices for the sequences in this split.
            # We'll reload the CSV index filter just to be sure.
            # (Note: this is only needed if dori didn't already have them).
            df = pd.read_csv(self.file_path)
            df['ori_index'] = df.index
            # Convert self.folds to list if it's a set
            fold_list = list(self.folds) if isinstance(self.folds, (set, range)) else self.folds
            filtered_df = df[df["fold"].isin(fold_list)]
            indices = filtered_df['ori_index'].values
            
            if len(indices) != len(self.dori["rseqs"]):
                print(f"Warning: Sequence count mismatch! CSV={len(indices)}, Dataloader={len(self.dori['rseqs'])}")
                # Dynamic fallback: if counts match, assume order is same
                if all_target_l0.shape[0] == len(self.dori["rseqs"]):
                     self.dori["target_l0"] = all_target_l0
                     self.dori["target_t"] = all_target_t
                else:
                    self.dori["target_l0"] = torch.zeros_like(self.dori["rseqs"])
                    self.dori["target_t"] = torch.zeros_like(self.dori["rseqs"])
            else:
                self.dori["target_l0"] = all_target_l0[indices]
                self.dori["target_t"] = all_target_t[indices]
            print(f"Aligned {len(self.dori['target_l0'])} targets.")
        else:
            self.dori["target_l0"] = torch.zeros_like(self.dori["rseqs"])
            self.dori["target_t"] = torch.zeros_like(self.dori["rseqs"])

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Standard loader, but we store file_path and folds for alignment.
        """
        self.file_path = sequence_path
        self.folds = folds
        # Call parent implementation to maintain compatibility with its pickle system
        return super().__load_data__(sequence_path, folds, pad_val)

    def __getitem__(self, index):
        # We need to temporarily remove non-sequence keys from dori 
        # so KTDataset.__getitem__ doesn't try to slice them.
        saved_items = {}
        non_seq_keys = ["target_l0", "target_t", "ori_row"]
        for k in non_seq_keys:
            if k in self.dori:
                saved_items[k] = self.dori.pop(k)
        
        try:
            dcur = super().__getitem__(index)
        finally:
            # Restore keys
            for k, v in saved_items.items():
                self.dori[k] = v

        # Add targets to the shift-style batch
        mseqs = self.dori["masks"][index]
        target_l0 = self.dori["target_l0"][index][1:] * mseqs
        target_t = self.dori["target_t"][index][1:] * mseqs
        
        dcur["target_l0"] = target_l0
        dcur["target_t"] = target_t
        
        return dcur
