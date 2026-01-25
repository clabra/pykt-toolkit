import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class GTransformer(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff, 
            kq_same, final_fc_dim, n_heads, separate_qa, l2_rasch, emb_type, emb_path, pretrain_dim, ablation, n_uid, **kwargs):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "gtransformer"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2_rasch = l2_rasch
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.ablation = ablation
        self.n_uid = n_uid
        # Personalization flag: controls whether student-specific parameters are enabled
        # Derived from n_uid for backward compatibility
        self.personalization = (n_uid > 0)
        
        # Loss component weights (Passed through from data_config/params)
        self.lambda_sup = kwargs.get('lambda_sup', 1.0)
        self.lambda_initmastery = kwargs.get('lambda_initmastery', 0.1)
        self.lambda_rate = kwargs.get('lambda_rate', 0.1)
        self.lambda_ref = kwargs.get('lambda_ref', 0.5)
        self.lambda_probe = kwargs.get('lambda_probe', 1.0)

        embed_l = d_model
        
        # Step 2 & 3: Parameter Projection Layers and Texture
        # z context vector is of dimension d_model + embed_l (concat of decoder output and question embedding)
        z_dim = d_model + embed_l
        
        # 1. Standard AKT/GTransformer
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) # Problem difficulty (u_q)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # Difficulty variation across concepts (d_ct)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # Interaction variation (f_ct,rt)
        
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l) # interaction emb
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
                
        # 2. Grounded Embeddings (Only if ablation != "all")
        if self.ablation != "all":
            # Note: We use size 1 for bases (scalar logit) and size z_dim for axes (projection direction)
            
            # Relational Axes (Step 3: Texture)
            # Directions in latent space corresponding to "More Knowledgeable" or "Faster Learner"
            # Dimensions must match z_context (d_model + embed_l)
            self.knowledge_axis_emb = nn.Embedding(self.n_question + 1, z_dim) 
            self.velocity_axis_emb = nn.Embedding(self.n_question + 1, z_dim)
            nn.init.normal_(self.knowledge_axis_emb.weight, mean=0.0, std=0.02) # Axis implies direction, centered at 0
            nn.init.normal_(self.velocity_axis_emb.weight, mean=0.0, std=0.02) # Axis implies direction, centered at 0
            
            # Theoretical Bases (Grounding points from BKT)
            # These remain scalar logits
            self.l0_base_emb = nn.Embedding(self.n_question + 1, 1) # L0_skill (Prior Base)
            self.t_base_emb = nn.Embedding(self.n_question + 1, 1)  # T_skill (Velocity Base)
            
            # Buffers for population-level parameters (Guess/Slip)
            # These are loaded from pre-fit BKT and kept constant during Reference Output generation
            self.register_buffer('bkt_guess', torch.ones(n_question + 1) * 0.2)
            self.register_buffer('bkt_slip', torch.ones(n_question + 1) * 0.1)
            self.register_buffer('bkt_l0_pop', torch.ones(n_question + 1) * 0.5)
            self.register_buffer('bkt_t_pop', torch.ones(n_question + 1) * 0.1)
            
            if self.personalization:
                self.student_param = nn.Embedding(self.n_uid + 1, 1) # Student learning velocity scalar (v_s)
                self.student_gap_param = nn.Embedding(self.n_uid + 1, 1) # Student knowledge gap scalar (k_c)
            
            # Phase 2: Probe Architecture
            # Dedicated shared linear probe heads for Active Grounding
            self.probe_l0 = nn.Linear(z_dim, 1)
            self.probe_t = nn.Linear(z_dim, 1)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

        # Step 2: Parameter Projection Layers (The "Grounded Outputs")
        # z context vector is of dimension d_model + embed_l (concat of decoder output and question embedding)
        # z_dim = d_model + embed_l # Moved up
        if self.ablation != "all":
            # p_L0 and p_T Grounded Outputs (Projecting z context vector)
            self.register_buffer('bkt_l0_pop', torch.ones(n_question + 1) * 0.5)
            self.register_buffer('bkt_t_pop', torch.ones(n_question + 1) * 0.1)

        self.out = nn.Sequential(
            nn.Linear(z_dim,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def load_theory_params(self, bkt_skill_params):
        """
        Initialize l0_base_emb and t_base_emb with pre-calculated BKT parameters.
        Using Textured Grounding: Embeddings are shifted logits with feature variance
        to survive LayerNorm blocks.
        """
        # If ablation="all", we do NOT use theory params, so we skip initialization.
        if self.ablation == "all":
            return
            
        if bkt_skill_params is None:
            return
        
        # Extract params mapping
        params_dict = bkt_skill_params.get('params', {})
        
        # Determine global fallbacks
        tmp_global = bkt_skill_params.get('global', {})
        if isinstance(tmp_global, dict) and 'prior' in tmp_global:
            global_params = tmp_global
        else:
            # Fallback: calculate mean from available skill params
            if params_dict:
                # Handle potential array wrappers in pyBKT values
                def get_val(d, k, def_val):
                    v = d.get(k, def_val)
                    if isinstance(v, (np.ndarray, list)):
                        return float(v[0])
                    return float(v)

                all_priors = [get_val(p, 'prior', 0.5) for p in params_dict.values()]
                all_learns = [get_val(p, 'learns', 0.1) for p in params_dict.values()]
                global_params = {
                    'prior': np.mean(all_priors) if all_priors else 0.5,
                    'learns': np.mean(all_learns) if all_learns else 0.1
                }
            else:
                global_params = {'prior': 0.5, 'learns': 0.1}
        
        def to_logit(p, eps=1e-6):
            p = np.clip(p, eps, 1.0 - eps)
            return np.log(p / (1.0 - p))

        with torch.no_grad():
            for q_idx in range(self.n_question + 1):
                # Try integer key, then string key, then fallback to global
                s_params = params_dict.get(q_idx, params_dict.get(str(q_idx), global_params))
                
                # Retrieve values, handling possible array/scalar types from pyBKT
                def extract(d, k, fallback_d, fallback_k):
                    v = d.get(k, fallback_d.get(fallback_k, 0.5))
                    if isinstance(v, (np.ndarray, list)):
                        return float(v[0])
                    return float(v)

                l0_p = extract(s_params, 'prior', global_params, 'prior')
                t_p = extract(s_params, 'learns', global_params, 'learns')
                
                # Textured Grounding: 
                # Instead of a constant vector, we use a small normal distribution 
                # centered at the logit. This ensures non-zero variance per-student
                # so LayerNorm doesn't zero out the features.
                l0_logit = to_logit(l0_p)
                t_logit = to_logit(t_p)
                
                # N(logit, 0.05)
                # Ensure the parameters exist (they might not if self.ablation="all", but we checked above)
                if hasattr(self, 'l0_base_emb'):
                    self.l0_base_emb.weight[q_idx].normal_(mean=l0_logit, std=0.05)
                if hasattr(self, 't_base_emb'):
                    self.t_base_emb.weight[q_idx].normal_(mean=t_logit, std=0.05)

                # Store population-level parameters for Reference Output
                self.bkt_l0_pop[q_idx] = l0_p
                self.bkt_t_pop[q_idx] = t_p
                self.bkt_guess[q_idx] = s_params.get('guess', 0.2)
                self.bkt_slip[q_idx] = s_params.get('slip', 0.1)
            
            # Initialize axes with orthogonal vectors for maximum initial diversity
            # This helps prevent collapse, but diversity loss is still needed to maintain it
            if hasattr(self, 'knowledge_axis_emb'):
                nn.init.orthogonal_(self.knowledge_axis_emb.weight)
            if hasattr(self, 'velocity_axis_emb'):
                nn.init.orthogonal_(self.velocity_axis_emb.weight)
                
        print(f"  [GTransformer] Textured Theory Bases (N(logit, 0.05)), Relational Axes, and Population Parameters initialized.")

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, uid_data=None, qtest=False):
        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2_rasch # rasch部分loss
        else:
            c_reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        # BS, seqlen, z_dim (d_model + embed_l)
        z_context = torch.cat([d_output, q_embed_data], dim=-1)
        
        # 1. Supervised Output (Standard AKT logic)
        output = self.out(z_context).squeeze(-1)
        preds = torch.sigmoid(output)

        if self.ablation == "all":
            outputs = {'predictions': preds}
            if not qtest:
                return outputs, c_reg_loss
            else:
                return outputs, c_reg_loss, z_context

        # 2. Step 2 & 3: Grounded Outputs via Semantic Axis Projection
        # Get concept-specific axes and bases
        # q_data: BS, seqlen
        k_axis = self.knowledge_axis_emb(q_data) # BS, seqlen, z_dim
        v_axis = self.velocity_axis_emb(q_data)  # BS, seqlen, z_dim
        
        l0_base = self.l0_base_emb(q_data).squeeze(-1) # BS, seqlen
        t_base = self.t_base_emb(q_data).squeeze(-1)   # BS, seqlen
        
        # Projection: Base + (z . Axis)
        # z_context: BS, seqlen, z_dim
        # Dot product along dim -1
        l0_logits = l0_base + (z_context * k_axis).sum(dim=-1)
        t_logits = t_base + (z_context * v_axis).sum(dim=-1)
        
        # Diversity Loss: Encourage semantic axes to be different across concepts
        # This prevents all axes from collapsing to identical vectors
        # Only apply when theory-guided mode is active (ablation != "all")
        if self.ablation != "all":
            unique_concepts = torch.unique(q_data)
            if len(unique_concepts) > 1 and not qtest:  # Only during training
                # Get axes for unique concepts in this batch
                sampled_k_axes = self.knowledge_axis_emb(unique_concepts)  # [N_unique, z_dim]
                sampled_v_axes = self.velocity_axis_emb(unique_concepts)  # [N_unique, z_dim]
                
                # Normalize to unit vectors for cosine similarity
                k_normalized = sampled_k_axes / (sampled_k_axes.norm(dim=1, keepdim=True) + 1e-8)
                v_normalized = sampled_v_axes / (sampled_v_axes.norm(dim=1, keepdim=True) + 1e-8)
                
                # Compute pairwise cosine similarities (should be low for diversity)
                k_sim_matrix = k_normalized @ k_normalized.t()  # [N_unique, N_unique]
                v_sim_matrix = v_normalized @ v_normalized.t()  # [N_unique, N_unique]
                
                # Penalize high off-diagonal similarities (we want orthogonal axes)
                # Mask out diagonal (self-similarity = 1.0)
                mask = ~torch.eye(len(unique_concepts), dtype=torch.bool, device=q_data.device)
                
                # Mean absolute cosine similarity (want this near 0)
                k_diversity_loss = k_sim_matrix[mask].abs().mean()
                v_diversity_loss = v_sim_matrix[mask].abs().mean()
                
                diversity_loss = 0.1 * (k_diversity_loss + v_diversity_loss)  # Weight to maintain orthogonality
            else:
                diversity_loss = torch.tensor(0.0, device=q_data.device)
        else:
            diversity_loss = torch.tensor(0.0, device=q_data.device)
        
        # Step 4: Individualization (Student-Specific scalars)
        # Adds static student bias to the dynamic estimate
        if self.personalization and uid_data is not None:
             # uid_data: BS (assumed constant per sequence or BS, seqlen if available)
             # Expand to match sequence if necessary
             if uid_data.dim() == 1:
                 uid_seq = uid_data.unsqueeze(1).expand(-1, q_data.size(1))
             else:
                 uid_seq = uid_data
                 
             # Get student params: [BS, seqlen]
             s_gap = self.student_gap_param(uid_seq).squeeze(-1)
             s_vel = self.student_param(uid_seq).squeeze(-1)
             
             l0_logits = l0_logits + s_gap
             t_logits = t_logits + s_vel
        
        p_l0 = torch.sigmoid(l0_logits) # BS, seqlen
        p_t = torch.sigmoid(t_logits)   # BS, seqlen

        # Phase 2: Probe Outputs
        # Compute probes for validation and active grounding
        probe_l0_logits = self.probe_l0(z_context).squeeze(-1) # BS, seqlen
        probe_t_logits = self.probe_t(z_context).squeeze(-1)   # BS, seqlen
        
        p_l0_probe = torch.sigmoid(probe_l0_logits)
        p_t_probe = torch.sigmoid(probe_t_logits)

        # Reference Output Generation via BKT Implementation
        # We process the batch but since each timestep t uses a different p_l0_t and p_t_t
        # were each prediction is the result of a BKT "walk" from 1 to t.
        
        # Initial Mastery State (L1) is the p_L0 generated at the current prediction timestep
        # We implement this vectorized to maintain performance
        ref_preds = self._bkt_ref_output(q_data, target, p_l0, p_t)

        # Collect all outputs in a structured dictionary
        outputs = {
            'predictions': preds,
            'p_l0': p_l0,
            'p_t': p_t,
            'p_l0_probe': p_l0_probe,
            'p_t_probe': p_t_probe,
            'reference_preds': ref_preds,   # Reference Output (BKT Logic Wrapper)
        }

        # Add diversity loss to regularization
        total_reg_loss = c_reg_loss + diversity_loss

        if not qtest:
            return outputs, total_reg_loss
        else:
            return outputs, total_reg_loss, z_context

    def _bkt_ref_output(self, q_data, target, p_l0, p_t):
        """
        Implements the BKT Logic Wrapper.
        For each timestep t, generates a prediction P(Y_t=1) by walking through
        history 1:t-1 using context-aware parameters p_l0(t) and p_t(t).
        """
        bs, seqlen = q_data.size()
        
        # Retrieve skill-specific population Guess and Slip
        # q_data: BS, seqlen
        gs = self.bkt_guess[q_data.long()] # BS, seqlen
        ss = self.bkt_slip[q_data.long()]  # BS, seqlen
        
        # Prepare for recurrence
        # We need to calculate mastery for every step.
        # But critically: a prediction at time t uses the parameters p_l0[t] and p_t[t]
        # and walks through the history of responses target[0...t-1].
        
        # To avoid O(T^2) code, we can optimize if parameters were constant, 
        # but since they are dynamic (contextual), we perform a masked walk.
        # However, for the Reference Output used in training/eval, 
        # it is common to use the parameters generated at index t to "re-interpret"
        # the journey that led to t.
        
        # Vectorized BKT Walk (Across entire batch and sequence)
        # current_L will hold the mastery belief.
        # Initialize with p_l0
        current_L = p_l0 # BS, seqlen
        
        # We iterate over the maximum sequence length to perform the Bayesian updates
        # ref_preds = []
        
        # Pre-calculate Bayes Updates for all possible L and target
        # L_post if corrected = L*(1-s) / (L*(1-s) + (1-L)*g)
        # L_post if incorrect = L*s / (L*s + (1-L)*(1-g))
        
        # Implementation Note: 
        # In GTransformer, z_t at time t already contains information about target[0...t-1].
        # The BKT wrapper's mission is to provide the "pedagogical constraint" head.
        
        # For efficiency, we implement the recurrence.
        # Start with Mastery for t=1
        all_masteries = []
        l_step = p_l0[:, 0] # Initial knowledge for the start of the sequence
        
        # Wait, if p_l0 is dynamic, which one do we use?
        # Following the architecture: p_l0[t] is the estimate of the student's 
        # starting point given evidence up to t.
        # Thus, for the prediction at index t, we use p_l0[t] and p_t[t].
        
        # This requires T steps of BKT for each of the T timesteps? 
        # Yes, to be strictly correct with "contextual parameters".
        # But we can simplify: we can use the latest parameters p_l0[seqlen-1] 
        # to retrospective describe the whole sequence, or use p_l0[t] and p_t[t]
        # to generate a single prediction at t.
        
        # Strategy: Use p_l0[t] and p_t[t] to predict y[t] given journey y[0:t-1].
        # Mastery at t=0 is p_l0[t].
        # Loop i from 0 to t-1: Update mastery with target[i].
        # Final mastery after walk is L_t.
        # Prediction is L_t*(1-s) + (1-L_t)*g.

        # Optimized vectorized version for sequence:
        # We can't avoid the inner loop easily if parameters are dynamic per timestep t.
        # But we can vectorize the BKT formulas!
        
        # Let's perform a "cumulative" BKT walk where at each step i, 
        # we calculate updates for ALL possible future contexts t.
        
        # shape: BS, seqlen (i), seqlen (t) - where i is the history step and t is the context
        # This might be memory intensive (BS*200*200). 
        # For seqlen=200, BS=64, it's 2.5M floats per level. Very doable.
        
        # current_L_retrospective: BS, seqlen_t (context), seqlen_i (history)
        L = p_l0.unsqueeze(-1).expand(bs, seqlen, seqlen).clone() # Initialized with p_l0[t] for all history i
        # NOTE: Removed static T_rate expansion - now using time-varying p_t[i] in loop
        
        # Iterative update for history i
        for i in range(seqlen - 1):
            # Mastery belief at history step i: L[:, :, i]
            # Observation at history step i: target[:, i]
            # We want to update Mastery for history step i+1: L[:, :, i+1]
            
            # 1. Bayes Update (Post-observation)
            obs = target[:, i].view(bs, 1, 1).expand(bs, seqlen, 1) # History evidence at i
            
            # Skill params at history step i
            g_i = gs[:, i].view(bs, 1, 1).expand(bs, seqlen, 1)
            s_i = ss[:, i].view(bs, 1, 1).expand(bs, seqlen, 1)
            
            L_i = L[:, :, i:i+1] # BS, seqlen_t, 1
            
            # Likelihoods
            prob_correct = L_i * (1 - s_i) + (1 - L_i) * g_i
            # P(L|Y=1)
            L_post_1 = (L_i * (1 - s_i)) / torch.clamp(prob_correct, min=1e-6)
            # P(L|Y=0)
            L_post_0 = (L_i * s_i) / torch.clamp(1 - prob_correct, min=1e-6)
            
            L_post = torch.where(obs > 0.5, L_post_1, L_post_0)
            
            # 2. Learning Transition (TIME-VARYING FIX)
            # Use HISTORICAL transition rate p_t[i] at this specific timestep
            # Shape: p_t[:, i:i+1] = [BS, 1] -> unsqueeze -> [BS, 1, 1] -> expand -> [BS, seqlen_t, 1]
            T_rate_i = p_t[:, i:i+1].unsqueeze(-1).expand(bs, seqlen, 1)
            
            # Validation: Check tensor shapes (only on first iteration to avoid overhead)
            if i == 0:
                assert T_rate_i.shape == (bs, seqlen, 1), f"T_rate_i shape mismatch: {T_rate_i.shape} vs expected ({bs}, {seqlen}, 1)"
                assert L_post.shape == (bs, seqlen, 1), f"L_post shape mismatch: {L_post.shape}"
            
            L_next = L_post + (1 - L_post) * T_rate_i
            
            # Fill the next history step for all contexts
            # But only for contexts t > i (where this history is relevant)
            # However, filling the whole block is faster in Torch
            L[:, :, i+1:i+2] = L_next

        # Now, for each context t, the mastery for the prediction at t is L[:, t, t]
        # Diagonal extraction
        idx = torch.arange(seqlen).to(device)
        L_at_t = L[:, idx, idx] # BS, seqlen
        
        # Final Output Emission
        # Note: we use Guess/Slip for the skill at index t
        ref_preds = L_at_t * (1 - ss) + (1 - L_at_t) * gs
        
        return ref_preds


class Architecture(nn.ModuleList):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
            for _ in range(n_blocks*2)
        ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data) # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data) # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.emb_type = emb_type
        if emb_type.endswith("avgpool"):
            # pooling
            #self.pool =  nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            pool_size = 3
            self.pooling =  nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.endswith("linear"):
            # linear
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)
            self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            # v = v.transpose(1,2)
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)#.contiguous().view(bs, -1, self.d_model)
        elif self.emb_type.endswith("linear"):
            # v = v.transpose(1,2)
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)
        elif self.emb_type.startswith("qid"):
            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * d_model

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention using function we will define next
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) # 所有v后置一位
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        # print(f"distotal_scores: {disttotal_scores}")
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    if pdiff == None:
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
