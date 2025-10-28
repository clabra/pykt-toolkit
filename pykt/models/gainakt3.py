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
                 peer_alignment_weight: float = 0.0,
                 difficulty_ordering_weight: float = 0.0,
                 drift_smoothness_weight: float = 0.0,
                 attempt_confidence_k: float = 10.0,
                 warmup_constraint_epochs: int = 0,
                 gate_init_bias: float = -2.0,
                 use_peer_context: bool = False,
                 use_difficulty_context: bool = False,
                 disable_peer: bool = False,
                 disable_difficulty: bool = False,
                 disable_fusion_broadcast: bool = False,
                 broadcast_last_context: bool = False,  # default False: preserve temporal dynamics (broadcast hurts AUC & interpretability)
                 disable_difficulty_penalty: bool = False,
                 fusion_for_heads_only: bool = True,
                 device: Optional[str] = None):
        super().__init__()
        # Core hyperparameters
        self.num_c = num_c
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.d_ff = d_ff
        self.dropout = dropout
        self.emb_type = emb_type
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
        self.peer_alignment_weight = peer_alignment_weight
        self.difficulty_ordering_weight = difficulty_ordering_weight
        self.drift_smoothness_weight = drift_smoothness_weight
        self.attempt_confidence_k = attempt_confidence_k
        self.warmup_constraint_epochs = warmup_constraint_epochs
        self.gate_init_bias = gate_init_bias
        self.current_epoch = 0  # externally adjustable by trainer
        self.use_peer_context = use_peer_context
        self.use_difficulty_context = use_difficulty_context
        self.disable_peer = disable_peer
        self.disable_difficulty = disable_difficulty
        # Architectural toggle semantics:
        #   broadcast_last_context = True  => collapse temporal context to last fused state (historically hurt AUC)
        #   broadcast_last_context = False => preserve per-step temporal context (baseline)
        # Legacy disable_fusion_broadcast flag (True => per-step) retained; it overrides only if
        # broadcast_last_context not explicitly set True.
        if broadcast_last_context and disable_fusion_broadcast:
            # Conflicting user intent: prefer explicit new flag and inform via debug print once.
            if os.environ.get('GAINAKT3_ARCH_DEBUG', '0') == '1':
                print('[GainAKT3] Both broadcast_last_context=True and disable_fusion_broadcast=True provided; using broadcast_last_context.')
        if disable_fusion_broadcast and not broadcast_last_context:
            broadcast_last_context = False  # explicit for clarity
        self.broadcast_last_context = broadcast_last_context
        self.disable_fusion_broadcast = disable_fusion_broadcast
        self.disable_difficulty_penalty = disable_difficulty_penalty
        self.fusion_for_heads_only = fusion_for_heads_only

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
        # Initialize gate bias negatively to start with closed gates (stability)
        if self.fusion_gate.bias is not None:
            nn.init.constant_(self.fusion_gate.bias, gate_init_bias)

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
        # ---- Defensive range & integrity checks (avoid opaque CUDA device-side asserts) ----
        if os.environ.get('GAINAKT3_INDEX_DEBUG', '0') == '1':
            # Lightweight index checks (non-fatal unless strict env set)
            try:
                q_min = int(q.min().item())
                q_max = int(q.max().item())
                r_min = int(r_int.min().item())
                r_max = int(r_int.max().item())
                if (r_min < 0) or (r_max > 1):
                    print(f"[GainAKT3][INDEX-ERROR] r outside {{0,1}}: range=[{r_min},{r_max}]")
                if (q_min < 0) or (q_max >= self.num_c):
                    print(f"[GainAKT3][INDEX-ERROR] q out-of-range: range=[{q_min},{q_max}] num_c={self.num_c}")
            except Exception as _dbg_e:
                if os.environ.get('GAINAKT3_INDEX_DEBUG_STRICT','1') == '1':
                    raise
                else:
                    print(f"[GainAKT3][INDEX-WARN] Lightweight index check failed: {_dbg_e}")
        positions = pos_encode(L).to(self.device)
        ctx = self.context_embedding(interaction_tokens) + self.pos_embedding(positions)
        val = self.value_embedding(interaction_tokens) + self.pos_embedding(positions)

        # Causal upper-triangular attention mask (future positions blocked = True above diagonal).
        mask = ut_mask(L).to(q.device)
        if not hasattr(self, '_mask_debugged'):
            self._mask_debugged = False
        if mask.dtype is not torch.bool:
            print(f"[GainAKT3] WARN: mask arrived as dtype {mask.dtype}; converting to bool explicitly.")
            mask = mask.bool()
        if (not self._mask_debugged) and os.environ.get('GAINAKT3_MASK_DEBUG','0') == '1':
            print(f"[GainAKT3] DEBUG mask dtype={mask.dtype}, shape={tuple(mask.shape)}, true_fraction={mask.float().mean().item():.4f}")
            self._mask_debugged = True
        import warnings as _w
        if os.environ.get('GAINAKT3_SUPPRESS_MASK_WARN','1') == '1':
            with _w.catch_warnings():
                _w.filterwarnings('ignore', message='Converting mask without torch.bool dtype to bool')
                ctx = self.context_encoder(ctx, mask)
                val = self.value_encoder(val, mask)
        else:
            ctx = self.context_encoder(ctx, mask)
            val = self.value_encoder(val, mask)

        # Peer vector retrieval (optional)
        if self.use_peer_context and (not self.disable_peer):
            if (not self.cold_start) and self.peer_index and 'peer_state_cluster_centroids' in self.peer_index:
                centroids = self.peer_index['peer_state_cluster_centroids'][: self.peer_K]
                if centroids is not None and len(centroids) > 0:
                    centroids_tensor = torch.tensor(centroids, dtype=ctx.dtype, device=ctx.device)
                    base_vec = ctx[:, -1, :].unsqueeze(1)
                    sims = F.cosine_similarity(base_vec, centroids_tensor.unsqueeze(0), dim=-1)
                    weights = F.softmax(self.peer_similarity_gamma * sims, dim=-1)
                    peer_vec = (weights.unsqueeze(-1) * centroids_tensor.unsqueeze(0)).sum(dim=1)
                else:
                    peer_vec = torch.zeros(B, self.d_model, device=ctx.device)
            else:
                peer_vec = torch.zeros(B, self.d_model, device=ctx.device)
        else:
            peer_vec = torch.zeros(B, self.d_model, device=ctx.device)

        # Difficulty vector (currently simple last value state) optionally used
        if self.use_difficulty_context and (not self.disable_difficulty):
            difficulty_vec = val[:, -1, :]
        else:
            difficulty_vec = torch.zeros(B, self.d_model, device=ctx.device)

        # Fusion (residual augmentation of last context state) with ablation capability
        h_last = ctx[:, -1, :]
        gate_in = torch.cat([h_last, peer_vec, difficulty_vec], dim=-1)
        gates = torch.sigmoid(self.fusion_gate(gate_in))
        g_p, g_d = gates[:, 0].unsqueeze(-1), gates[:, 1].unsqueeze(-1)
        augmented_h = h_last + g_p * peer_vec + g_d * difficulty_vec
        augmented_h = F.layer_norm(augmented_h, (self.d_model,))

    # (fusion_for_heads_only kept for backward compatibility; actual selection controlled by broadcast_last_context)

        # Prediction
        target_concepts = q if qry is None else qry
        target_emb = self.concept_embedding(target_concepts)
        # Decide predictor input depending on architectural flag
        if self.broadcast_last_context:
            concat_ctx = augmented_h.unsqueeze(1).expand(-1, L, -1)
        else:
            concat_ctx = ctx
        concat = torch.cat([concat_ctx, val, target_emb], dim=-1)
        base_logits = self.prediction_head(concat).squeeze(-1)
        difficulty_logit = self.difficulty_head(difficulty_vec).squeeze(-1)
        # Sequence difficulty logits for drift (context-based) - auxiliary only
        difficulty_logits_seq = self.difficulty_head(ctx).squeeze(-1)  # [B,L]
        if self.disable_difficulty_penalty:
            logits = base_logits  # skip subtraction for ablation
            difficulty_penalty_contrib = torch.zeros_like(base_logits)
        else:
            logits = base_logits - self.beta_difficulty * difficulty_logit.unsqueeze(1)
            difficulty_penalty_contrib = - self.beta_difficulty * difficulty_logit.unsqueeze(1)
        preds = torch.sigmoid(logits)

        # ---------------- Decomposition Components ----------------
        # Separate prediction_head weight into 3 segments (prediction_ctx, value, target_emb)
        W_full = self.prediction_head.weight.view(-1)  # [3*d_model]
        W_h = W_full[: self.d_model]
        W_v = W_full[self.d_model: 2 * self.d_model]
        W_t = W_full[2 * self.d_model:]
        bias_term = self.prediction_head.bias.detach() if self.prediction_head.bias is not None else torch.tensor(0.0, device=ctx.device)

        # Decompose augmented_h into intrinsic + peer + difficulty gated parts
        h_intrinsic = h_last  # original context
        h_peer_component = g_p * peer_vec    # [B,d]
        h_diff_component = g_d * difficulty_vec  # [B,d]

        # Broadcast components over sequence length
        h_intrinsic_seq = h_intrinsic.unsqueeze(1).expand(-1, L, -1)
        h_peer_seq = h_peer_component.unsqueeze(1).expand(-1, L, -1)
        h_diff_seq = h_diff_component.unsqueeze(1).expand(-1, L, -1)

        mastery_contrib = (h_intrinsic_seq * W_h.unsqueeze(0).unsqueeze(0)).sum(-1)
        peer_prior_contrib = (h_peer_seq * W_h.unsqueeze(0).unsqueeze(0)).sum(-1)
        difficulty_fused_contrib = (h_diff_seq * W_h.unsqueeze(0).unsqueeze(0)).sum(-1)
        value_stream_contrib = (val * W_v.unsqueeze(0).unsqueeze(0)).sum(-1)
        concept_contrib = (target_emb * W_t.unsqueeze(0).unsqueeze(0)).sum(-1)
        bias_contrib = bias_term.view(1,1).expand_as(base_logits)
        # difficulty_penalty_contrib already defined above respecting ablation flag
        reconstructed_base = mastery_contrib + peer_prior_contrib + difficulty_fused_contrib + value_stream_contrib + concept_contrib + bias_contrib
        reconstruction_error = (base_logits - reconstructed_base).abs().mean().detach()

        out = {
            'predictions': preds,
            # expose raw difficulty logit per final state (before scaling & broadcasting)
            'difficulty_logit': difficulty_logit.detach(),
            'difficulty_logits_seq': difficulty_logits_seq.detach(),
            # fusion gate raw values for interpretability (peer gate g_p, difficulty gate g_d)
            'fusion_gates': torch.cat([g_p, g_d], dim=-1).detach(),
            'decomposition': {
                'mastery_contrib': mastery_contrib.detach(),
                'peer_prior_contrib': peer_prior_contrib.detach(),
                'difficulty_fused_contrib': difficulty_fused_contrib.detach(),
                'value_stream_contrib': value_stream_contrib.detach(),
                'concept_contrib': concept_contrib.detach(),
                'bias_contrib': bias_contrib.detach(),
                'difficulty_penalty_contrib': difficulty_penalty_contrib.detach(),
                'reconstruction_error': reconstruction_error
            }
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
        out['augmented_h_last'] = augmented_h.detach()
        out['prediction_context_mode'] = 'broadcast_last' if self.broadcast_last_context else 'per_timestep'
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
        # Peer alignment loss (uses last-step mastery vs peer_correct_rate)
        if self.use_mastery_head and self.peer_alignment_weight > 0 and self.use_peer_context and (not self.cold_start) and self.peer_index is not None:
            if 'peer_correct_rate' in self.peer_index and isinstance(self.peer_index['peer_correct_rate'], dict):
                last_concepts = q[:, -1]
                last_mastery = out.get('projected_mastery', torch.sigmoid(self.mastery_head(ctx)))[:, -1, :]  # [B,C]
                gathered_mastery = torch.gather(last_mastery, 1, last_concepts.unsqueeze(-1)).squeeze(-1)  # [B]
                peer_rates = []
                confidences = []
                for cid in last_concepts.tolist():
                    pr = self.peer_index['peer_correct_rate'].get(int(cid), 0.5)
                    ac = self.peer_index.get('peer_attempt_count', {}).get(int(cid), 0)
                    peer_rates.append(pr)
                    # confidence scaling: ac/(ac + k)
                    confidences.append(ac / (ac + self.attempt_confidence_k))
                peer_rates_t = torch.tensor(peer_rates, device=q.device, dtype=gathered_mastery.dtype)
                conf_t = torch.tensor(confidences, device=q.device, dtype=gathered_mastery.dtype)
                peer_align_mse = ((gathered_mastery - peer_rates_t) ** 2 * conf_t).mean()
                constraint_losses['peer_alignment_loss'] = peer_align_mse * self.peer_alignment_weight
        # Difficulty ordering loss (batch-level pairwise ranking using last-step predictions)
        if self.difficulty_ordering_weight > 0:
            preds_last = preds[:, -1]  # [B]
            diff_last = difficulty_logit  # [B]
            Bsz = preds_last.size(0)
            if Bsz > 1:
                # Sample up to P pairs (P = min(64, combinations))
                P = min(64, Bsz * (Bsz - 1) // 2)
                # Generate all pairs indices then randomly pick P
                pairs = []
                for i in range(Bsz):
                    for j in range(i+1, Bsz):
                        pairs.append((i,j))
                import random as _rnd
                _rnd.shuffle(pairs)
                pairs = pairs[:P]
                margin = 0.0
                ranking_losses = []
                for i,j in pairs:
                    di = diff_last[i]
                    dj = diff_last[j]
                    pi = preds_last[i]
                    pj = preds_last[j]
                    if di > dj:  # harder i should have lower probability
                        ranking_losses.append(F.relu(margin - (pj - pi)))
                    elif dj > di:
                        ranking_losses.append(F.relu(margin - (pi - pj)))
                if ranking_losses:
                    ranking_loss = torch.stack(ranking_losses).mean()
                    constraint_losses['difficulty_ordering_loss'] = ranking_loss * self.difficulty_ordering_weight
        # Drift smoothness loss (second difference over sequence difficulty logits)
        if self.drift_smoothness_weight > 0 and difficulty_logits_seq.size(1) > 2:
            diff_seq = difficulty_logits_seq  # [B,L]
            second_diff = diff_seq[:,2:] - 2*diff_seq[:,1:-1] + diff_seq[:,:-2]
            drift_penalty = second_diff.abs().mean()
            constraint_losses['drift_smoothness_loss'] = drift_penalty * self.drift_smoothness_weight
        # Aggregate constraint loss (respect warmup)
        if constraint_losses and (self.current_epoch >= self.warmup_constraint_epochs):
            total_constraint = sum(constraint_losses.values())
        else:
            total_constraint = torch.tensor(0.0, device=q.device)
        out['constraint_losses'] = {k: v.detach() for k,v in constraint_losses.items()}
        out['total_constraint_loss'] = total_constraint
        # DataParallel gather safety: wrap pure Python bools into tensors so scatter_gather does not fail
        for k,v in list(out.items()):
            if isinstance(v, bool):
                out[k] = torch.tensor(float(v), device=q.device)
        # Also normalize nested dicts (e.g., decomposition) any bool values
        for k,v in list(out.items()):
            if isinstance(v, dict):
                for nk,nv in list(v.items()):
                    if isinstance(nv, bool):
                        v[nk] = torch.tensor(float(nv), device=q.device)
        # Normalize zero-dim tensors to shape (1,) to avoid DataParallel scalar gather warning
        def _reshape_scalars(container):
            if isinstance(container, dict):
                for kk, vv in container.items():
                    if torch.is_tensor(vv) and vv.ndim == 0:
                        container[kk] = vv.view(1)
            elif torch.is_tensor(container) and container.ndim == 0:
                return container.view(1)
            return container
        for kk in list(out.keys()):
            out[kk] = _reshape_scalars(out[kk])
        if 'constraint_losses' in out:
            out['constraint_losses'] = _reshape_scalars(out['constraint_losses'])
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
        peer_alignment_weight=config.get('peer_alignment_weight', 0.0),
        difficulty_ordering_weight=config.get('difficulty_ordering_weight', 0.0),
        drift_smoothness_weight=config.get('drift_smoothness_weight', 0.0),
        attempt_confidence_k=config.get('attempt_confidence_k', 10.0),
        warmup_constraint_epochs=config.get('warmup_constraint_epochs', 0),
        gate_init_bias=config.get('gate_init_bias', -2.0),
        disable_fusion_broadcast=config.get('disable_fusion_broadcast', False),
        disable_difficulty_penalty=config.get('disable_difficulty_penalty', False),
        fusion_for_heads_only=config.get('fusion_for_heads_only', True),
        broadcast_last_context=config.get('broadcast_last_context', False),
        device=config.get('device')
    )
