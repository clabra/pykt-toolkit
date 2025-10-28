"""GainAKT3 Model

Transformer-based Knowledge Tracing with peer similarity and historical difficulty context.

Notes:
 - Lives alongside existing models without modifying them.
 - External artifacts (peer index, difficulty table) are optional; cold_start mode if missing.
 - Forward signature aligns with PyKT expectations (q,r,qtest) returning predictions and optional mastery/gain projections.
"""
import os
import pickle
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .utils import ut_mask, pos_encode


def _safe_sha(path: str) -> str:
    if not os.path.exists(path):
        return 'MISSING'
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


class GainAKT3(nn.Module):
    def __init__(self,
                 num_c: int,
                 seq_len: int = 200,
                 d_model: int = 128,
                 n_heads: int = 8,
                 num_encoder_blocks: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 dataset: str = 'assist2015',
                 peer_K: int = 8,
                 peer_similarity_gamma: float = 4.0,
                 beta_difficulty: float = 1.0,
                 use_mastery_head: bool = True,
                 use_gain_head: bool = True,
                 emb_type: str = 'qid',
                 artifact_base: str = 'data',
                 # Auxiliary interpretability constraint weights
                 alignment_weight: float = 0.0,
                 sparsity_weight: float = 0.0,
                 consistency_weight: float = 0.0,
                 retention_weight: float = 0.0,
                 lag_gain_weight: float = 0.0,
                 warmup_constraint_epochs: int = 0,
                 device: Optional[str] = None):
        super().__init__()
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.peer_K = peer_K
        self.peer_similarity_gamma = peer_similarity_gamma
        self.beta_difficulty = beta_difficulty
        self.use_mastery_head = use_mastery_head
        self.use_gain_head = use_gain_head
        self.dataset = dataset
        self.artifact_base = artifact_base
        # Store auxiliary weights
        self.alignment_weight = alignment_weight
        self.sparsity_weight = sparsity_weight
        self.consistency_weight = consistency_weight
        self.retention_weight = retention_weight
        self.lag_gain_weight = lag_gain_weight
        self.warmup_constraint_epochs = warmup_constraint_epochs
        self.current_epoch = 0  # externally adjustable by trainer

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Embeddings
        self.context_embedding = nn.Embedding(num_c * 2, d_model)
        self.value_embedding = nn.Embedding(num_c * 2, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.concept_embedding = nn.Embedding(num_c, d_model)

        # Encoder blocks (dual-stream)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_blocks)
        self.value_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_blocks)

        # Heads
        self.prediction_head = nn.Linear(d_model * 3, 1)
        if use_mastery_head:
            self.mastery_head = nn.Linear(d_model, num_c)
        if use_gain_head:
            self.gain_head = nn.Linear(d_model, num_c)
        self.difficulty_head = nn.Linear(d_model, 1)

        # Fusion gate for peer & difficulty
        self.fusion_gate = nn.Linear(d_model * 3, 2)

        # Artifacts
        self.peer_path = os.path.join(artifact_base, 'peer_index', dataset, 'peer_index.pkl')
        self.diff_path = os.path.join(artifact_base, 'difficulty', dataset, 'difficulty_table.parquet')
        self.peer_index = self._load_pickle(self.peer_path)
        self.peer_hash = _safe_sha(self.peer_path)
        self.diff_hash = _safe_sha(self.diff_path)
        self.cold_start = (self.peer_index is None) or not os.path.exists(self.diff_path)

        self.to(self.device)

    def _load_pickle(self, path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def forward(self, q: torch.Tensor, r: torch.Tensor, qry: torch.Tensor = None, qtest: bool = False):
        # q,r: [B,L]
        # Ensure inputs on model device
        q = q.to(self.device)
        r = r.to(self.device)
        B, L = q.size()
        r_int = r.long()
        interaction_tokens = q + self.num_c * r_int
        positions = pos_encode(L).to(self.device)
        ctx = self.context_embedding(interaction_tokens) + self.pos_embedding(positions)
        val = self.value_embedding(interaction_tokens) + self.pos_embedding(positions)

        mask = ut_mask(L).to(q.device)
        ctx = self.context_encoder(ctx, mask)
        val = self.value_encoder(val, mask)

        # Peer vector
        if (not self.cold_start) and self.peer_index and 'peer_state_cluster_centroids' in self.peer_index:
            centroids = self.peer_index['peer_state_cluster_centroids'][: self.peer_K]
            if centroids:
                centroids_tensor = torch.tensor(centroids, dtype=ctx.dtype, device=ctx.device)
                base_vec = ctx[:, -1, :].unsqueeze(1)
                sims = F.cosine_similarity(base_vec, centroids_tensor.unsqueeze(0), dim=-1)
                weights = F.softmax(self.peer_similarity_gamma * sims, dim=-1)
                peer_vec = (weights.unsqueeze(-1) * centroids_tensor.unsqueeze(0)).sum(dim=1)
            else:
                peer_vec = torch.zeros(B, self.d_model, device=ctx.device)
        else:
            peer_vec = torch.zeros(B, self.d_model, device=ctx.device)

        # Difficulty vec (placeholder last value state)
        difficulty_vec = val[:, -1, :]

        # Fusion
        h_last = ctx[:, -1, :]
        gate_in = torch.cat([h_last, peer_vec, difficulty_vec], dim=-1)
        gates = torch.sigmoid(self.fusion_gate(gate_in))
        g_p, g_d = gates[:, 0].unsqueeze(-1), gates[:, 1].unsqueeze(-1)
        # (Optional fused representation was removed for cleanliness.)

        # Prediction
        target_concepts = q if qry is None else qry
        target_emb = self.concept_embedding(target_concepts)
        concat = torch.cat([ctx, val, target_emb], dim=-1)
        base_logits = self.prediction_head(concat).squeeze(-1)
        difficulty_logit = self.difficulty_head(difficulty_vec).squeeze(-1)
        logits = base_logits - self.beta_difficulty * difficulty_logit.unsqueeze(1)
        preds = torch.sigmoid(logits)

        out = {
            'predictions': preds,
            # expose raw difficulty logit per final state (before scaling & broadcasting)
            'difficulty_logit': difficulty_logit.detach(),
            # fusion gate raw values for interpretability (peer gate g_p, difficulty gate g_d)
            'fusion_gates': torch.cat([g_p, g_d], dim=-1).detach()
        }
        if qtest:
            out['encoded_seq'] = ctx
        if self.use_mastery_head:
            mastery_raw = self.mastery_head(ctx)
            mastery_proj = torch.sigmoid(mastery_raw)
            # expose both raw (pre-sigmoid) and projected mastery sequence
            out['mastery_raw'] = mastery_raw.detach()
            out['projected_mastery'] = mastery_proj
        if self.use_gain_head:
            gains_raw = self.gain_head(val)
            gains = torch.relu(gains_raw)
            out['gains_raw'] = gains_raw.detach()
            out['projected_gains'] = gains
        # Interpretability extras
        out['peer_influence_share'] = g_p.mean().detach()
        out['difficulty_adjustment_magnitude'] = (g_d.abs() * difficulty_logit.abs()).mean().detach()
        out['peer_hash'] = self.peer_hash
        out['difficulty_hash'] = self.diff_hash
        out['cold_start'] = self.cold_start
        # ---- Auxiliary losses (computed only if heads present & weights > 0) ----
        constraint_losses = {}
        if self.use_mastery_head and self.alignment_weight > 0:
            # Alignment: encourage mastery trajectory monotonic non-decreasing (soft)
            # penalize negative deltas
            mastery_deltas = mastery_proj[:,1:,:] - mastery_proj[:,:-1,:]
            alignment_penalty = F.relu(-mastery_deltas).mean()
            constraint_losses['alignment_loss'] = alignment_penalty * self.alignment_weight
        if self.use_gain_head and self.sparsity_weight > 0:
            # Sparsity: encourage many near-zero gains (L1 norm)
            sparsity_penalty = gains.abs().mean()
            constraint_losses['sparsity_loss'] = sparsity_penalty * self.sparsity_weight
        if self.use_mastery_head and self.use_gain_head and self.consistency_weight > 0:
            # Consistency: gains should correlate with mastery increments (positive correlation)
            mastery_inc = (mastery_proj[:,1:,:] - mastery_proj[:,:-1,:]).clamp(min=0)
            gains_pos = gains[:,1:,:]
            # Compute cosine similarity averaged
            eps = 1e-8
            sim = (mastery_inc * gains_pos).sum(-1) / ((mastery_inc.norm(dim=-1)+eps)*(gains_pos.norm(dim=-1)+eps))
            # penalty if similarity below threshold (target ~ 0.5)
            consistency_penalty = F.relu(0.5 - sim).mean()
            constraint_losses['consistency_loss'] = consistency_penalty * self.consistency_weight
        if self.use_mastery_head and self.retention_weight > 0:
            # Retention: discourage mastery decay (negative steps)
            mastery_deltas = mastery_proj[:,1:,:] - mastery_proj[:,:-1,:]
            retention_penalty = F.relu(-mastery_deltas).mean()
            constraint_losses['retention_loss'] = retention_penalty * self.retention_weight
        if self.use_gain_head and self.lag_gain_weight > 0:
            # Lag gain: encourage gains to precede mastery increases (temporal lead)
            # Use cross-correlation: gains(t) vs mastery_inc(t+1)
            if gains.size(1) > 2:
                mastery_inc_full = (mastery_proj[:,1:,:] - mastery_proj[:,:-1,:]).clamp(min=0)  # [B,L-1,C]
                # gains indices 0..L-2 should align with mastery_inc indices 1..L-2
                mastery_lead = mastery_inc_full[:,1:,:]        # [B,L-2,C]
                gains_trim = gains[:, :mastery_lead.size(1), :]  # truncate gains to L-2 positions
                eps = 1e-8
                lag_sim = (gains_trim * mastery_lead).sum(-1) / ((gains_trim.norm(dim=-1)+eps)*(mastery_lead.norm(dim=-1)+eps))
                lag_penalty = F.relu(0.3 - lag_sim).mean()
            else:
                lag_penalty = torch.tensor(0.0, device=q.device)
            constraint_losses['lag_gain_loss'] = lag_penalty * self.lag_gain_weight
        # Aggregate constraint loss (respect warmup)
        if constraint_losses and (self.current_epoch >= self.warmup_constraint_epochs):
            total_constraint = sum(constraint_losses.values())
        else:
            total_constraint = torch.tensor(0.0, device=q.device)
        out['constraint_losses'] = {k: v.detach() for k,v in constraint_losses.items()}
        out['total_constraint_loss'] = total_constraint
        return out


def create_gainakt3_model(config):
    return GainAKT3(
        num_c=config.get('num_c', 100),
        seq_len=config.get('seq_len', 200),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 8),
        num_encoder_blocks=config.get('num_encoder_blocks', 2),
        d_ff=config.get('d_ff', 256),
        dropout=config.get('dropout', 0.1),
        dataset=config.get('dataset', 'assist2015'),
        peer_K=config.get('peer_K', 8),
        peer_similarity_gamma=config.get('peer_similarity_gamma', 4.0),
        beta_difficulty=config.get('beta_difficulty', 1.0),
        use_mastery_head=config.get('use_mastery_head', True),
        use_gain_head=config.get('use_gain_head', True),
        emb_type=config.get('emb_type', 'qid'),
        artifact_base=config.get('artifact_base', 'data'),
        alignment_weight=config.get('alignment_weight', 0.0),
        sparsity_weight=config.get('sparsity_weight', 0.0),
        consistency_weight=config.get('consistency_weight', 0.0),
        retention_weight=config.get('retention_weight', 0.0),
        lag_gain_weight=config.get('lag_gain_weight', 0.0),
        warmup_constraint_epochs=config.get('warmup_constraint_epochs', 0),
        device=config.get('device')
    )
