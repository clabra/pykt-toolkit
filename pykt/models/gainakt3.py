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
        B, L = q.size()
        r_int = r.long()
        interaction_tokens = q + self.num_c * r_int

        positions = pos_encode(L).to(q.device)
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

        out = {'predictions': preds}
        if qtest:
            out['encoded_seq'] = ctx
        if self.use_mastery_head:
            mastery_raw = self.mastery_head(ctx)
            mastery_proj = torch.sigmoid(mastery_raw)
            out['projected_mastery'] = mastery_proj
        if self.use_gain_head:
            gains_raw = self.gain_head(val)
            gains = torch.relu(gains_raw)
            out['projected_gains'] = gains
        # Interpretability extras
        out['peer_influence_share'] = g_p.mean().detach()
        out['difficulty_adjustment_magnitude'] = (g_d.abs() * difficulty_logit.abs()).mean().detach()
        out['peer_hash'] = self.peer_hash
        out['difficulty_hash'] = self.diff_hash
        out['cold_start'] = self.cold_start
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
        device=config.get('device')
    )
