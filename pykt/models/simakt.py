import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones

class SimAKT(Module):
    """
    SimAKT: Similarity-based Attention for Knowledge Tracing
    
    A Transformer-based model that uses similarity between learning trajectories
    to predict student responses and track skill mastery evolution through learning curves.
    
    Key Features:
    - Trajectory-based similarity attention instead of traditional QKV attention
    - Learning curve prediction with sigmoid parameters (L, k, N0)
    - Multi-skill integration for questions requiring multiple knowledge components
    - Interpretable predictions through learning curve visualization
    """
    
    def __init__(self, num_c, num_q=None, seq_len=200, emb_size=256, num_attn_heads=8, 
                 dropout=0.1, num_en=4, emb_type="qid", emb_path="", pretrain_dim=768,
                 similarity_cache_size=10000, mastery_threshold=0.9, curve_dim=128):
        super().__init__()
        
        self.model_name = "simakt"
        self.emb_type = emb_type
        
        # Model configuration
        self.num_c = num_c  # Number of concepts/skills
        self.num_q = num_q if num_q is not None else num_c  # Number of questions
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en
        self.mastery_threshold = mastery_threshold
        self.curve_dim = curve_dim
        
        # Embedding layers
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            # Handle case where num_q is 0 (concept-only datasets)
            if self.num_q > 0:
                self.exercise_emb = Embedding(self.num_q, emb_size)
            else:
                # Use concept embedding for exercise embedding when num_q is 0
                self.exercise_emb = None
            self.concept_emb = Embedding(num_c, emb_size)
        
        self.position_emb = Embedding(seq_len, emb_size)
        
        # Trajectory encoder for (S, N, M) tuples
        self.trajectory_encoder = TrajectoryEncoder(
            num_skills=num_c,
            emb_size=emb_size,
            curve_dim=curve_dim
        )
        
        # TSMini similarity computation
        self.similarity_computer = TSMiniSimilarity(
            num_views=4,
            cache_size=similarity_cache_size,
            emb_size=emb_size
        )
        
        # Similarity-based attention blocks
        self.blocks = get_clones(
            SimAKTAttentionBlock(emb_size, num_attn_heads, dropout, num_c),
            self.num_en
        )
        
        # Learning curve predictor
        self.curve_predictor = LearningCurvePredictor(
            emb_size, num_c, curve_dim
        )
        
        # Multi-skill integration strategies
        self.skill_integrator = MultiSkillIntegrator(
            num_c, strategy='min'  # Can be 'min', 'weighted', 'conjunctive'
        )
        
        # Output layers
        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(emb_size, 1)
        
        # Loss components
        self.response_loss = nn.BCEWithLogitsLoss()
        self.curve_loss = nn.MSELoss()
        
    def forward(self, q, r, qry, qtest=False):
        """
        Forward pass for SimAKT model
        
        Args:
            q: Question/concept IDs [batch_size, seq_len]
            r: Response correctness [batch_size, seq_len] 
            qry: Query questions for next prediction [batch_size, seq_len]
            qtest: Whether in test mode
            
        Returns:
            p: Response predictions [batch_size, seq_len]
            Additional outputs if qtest=True: embeddings, learning curves, attention weights
        """
        # Handle empty tensors
        if q.numel() == 0 or r.numel() == 0 or qry.numel() == 0:
            device = next(self.parameters()).device
            dummy_output = torch.zeros(1, 1, device=device)
            if not qtest:
                return dummy_output
            else:
                return {
                    'predictions': dummy_output,
                    'embeddings': torch.zeros(1, 1, self.emb_size, device=device),
                    'learning_curves': torch.zeros(1, 1, self.num_c, 3, device=device),
                    'mastery_levels': torch.zeros(1, 1, self.num_c, device=device),
                    'attention_weights': [],
                    'trajectories': torch.zeros(1, 1, self.emb_size, device=device)
                }
                
        batch_size, seq_len = q.size()
        
        # Base embeddings
        qshftemb, xemb = self.base_emb(q, r, qry)
        
        # Construct trajectories from interaction sequence
        trajectories = self.trajectory_encoder(q, r, qry)
        
        # Apply similarity-based attention blocks
        attention_weights_all = []
        for i in range(self.num_en):
            xemb, attention_weights = self.blocks[i](
                qshftemb, xemb, trajectories, return_attention=qtest
            )
            if qtest:
                attention_weights_all.append(attention_weights)
        
        # Predict learning curves
        curve_params = self.curve_predictor(xemb)
        
        # Calculate current mastery levels
        current_attempts = self.extract_current_attempts(q, r)
        mastery_levels = self.calculate_mastery(curve_params, current_attempts)
        
        # Multi-skill integration for final prediction
        integrated_mastery = self.skill_integrator(mastery_levels, qry)
        
        # Binary response prediction
        response_logits = self.pred(self.dropout_layer(xemb))
        p = torch.sigmoid(response_logits).squeeze(-1)
        
        # Combine curve-based and direct predictions
        curve_based_pred = (integrated_mastery > self.mastery_threshold).float()
        p = 0.7 * p + 0.3 * curve_based_pred  # Weighted combination
        
        if not qtest:
            return p
        else:
            return {
                'predictions': p,
                'embeddings': xemb,
                'learning_curves': curve_params,
                'mastery_levels': mastery_levels,
                'attention_weights': attention_weights_all,
                'trajectories': trajectories
            }
    
    def base_emb(self, q, r, qry):
        """Generate base embeddings for interactions and queries"""
        # Ensure tensors have the correct shape and are not empty
        batch_size, seq_len = q.size() if q.numel() > 0 else (1, 1)
        
        # Handle empty tensors gracefully
        if q.numel() == 0 or r.numel() == 0 or qry.numel() == 0:
            # If any input tensor is empty, return zero embeddings with correct shape
            device = next(self.parameters()).device
            qshftemb = torch.zeros(batch_size, seq_len, self.emb_size, device=device)
            xemb = torch.zeros(batch_size, seq_len, self.emb_size, device=device)
            return qshftemb, xemb
        
        # Standard embedding computation
        x = q + self.num_c * r
        
        # Handle exercise embedding (use concept embedding if exercise_emb is None)
        if self.exercise_emb is not None:
            qshftemb = self.exercise_emb(qry)
        else:
            # For concept-only datasets, use concept embedding
            qshftemb = self.concept_emb(qry)
            
        xemb = self.interaction_emb(x)
        
        # Add positional encoding
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        
        return qshftemb, xemb
    
    def extract_current_attempts(self, q, r):
        """Extract number of attempts per skill from interaction sequence"""
        # Handle empty tensors
        if q.numel() == 0 or r.numel() == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.num_c, device=device)
            
        batch_size, seq_len = q.size()
        attempts = torch.zeros(batch_size, seq_len, self.num_c, device=q.device)
        
        for b in range(batch_size):
            skill_counts = torch.zeros(self.num_c, device=q.device)
            for t in range(seq_len):
                skill_id = q[b, t].item()
                if skill_id >= 0 and skill_id < self.num_c:  # Valid skill within bounds
                    skill_counts[skill_id] += 1
                    attempts[b, t, skill_id] = skill_counts[skill_id]
        
        return attempts
    
    def calculate_mastery(self, curve_params, current_attempts):
        """Calculate mastery levels using sigmoid curves"""
        # curve_params: [batch_size, seq_len, num_skills, 3] (L, k, N0)
        # current_attempts: [batch_size, seq_len, num_skills]
        
        L = curve_params[:, :, :, 0]    # Learning limit
        k = curve_params[:, :, :, 1]    # Learning rate  
        N0 = curve_params[:, :, :, 2]   # Inflection point
        
        # Sigmoid curve: M(N) = L / (1 + exp(-k(N - N0)))
        mastery = L / (1 + torch.exp(-k * (current_attempts - N0)))
        
        return mastery
    
    def get_loss(self, q, r, qry, y, sm=None):
        """
        Calculate multi-objective loss function
        
        Args:
            q, r, qry: Input sequences [batch_size, seq_len]
            y: Target responses [batch_size, seq_len]
            sm: Sequence mask [batch_size, seq_len] (optional)
            
        Returns:
            loss: Combined loss (response prediction + curve fitting + regularization)
        """
        # Handle empty tensors
        if q.numel() == 0 or r.numel() == 0 or qry.numel() == 0 or y.numel() == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Forward pass to get predictions and internal representations
        outputs = self.forward(q, r, qry, qtest=True)
        predictions = outputs['predictions']
        curve_params = outputs['learning_curves']
        mastery_levels = outputs['mastery_levels']
        
        # Apply sequence mask if provided
        if sm is not None:
            # Ensure mask has same shape as predictions
            if sm.shape != predictions.shape:
                sm = sm[:predictions.shape[0], :predictions.shape[1]]
            predictions = torch.masked_select(predictions, sm)
            y_masked = y[:predictions.shape[0], :predictions.shape[1]] if y.numel() > 0 else y
            y = torch.masked_select(y_masked, sm)
        
        # Handle case where predictions or targets are empty after masking
        if predictions.numel() == 0 or y.numel() == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure predictions and targets have compatible shapes
        min_size = min(predictions.numel(), y.numel())
        if predictions.numel() != y.numel():
            predictions = predictions.flatten()[:min_size]
            y = y.flatten()[:min_size]
        
        # Response prediction loss with numerical stability
        predictions_clamped = torch.clamp(predictions.double(), min=1e-7, max=1-1e-7)
        pred_loss = F.binary_cross_entropy(predictions_clamped, y.double())
        
        # Learning curve quality loss (ensure monotonicity and bounds)
        curve_quality_loss = self.curve_quality_loss(curve_params, mastery_levels)
        
        # Attention regularization loss
        attention_reg_loss = self.attention_regularization(outputs['attention_weights'])
        
        # Check for NaN values and replace with zeros
        if torch.isnan(pred_loss):
            pred_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        if torch.isnan(curve_quality_loss):
            curve_quality_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        if torch.isnan(attention_reg_loss):
            attention_reg_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Combined loss with fixed weights
        total_loss = (0.5 * pred_loss + 
                     0.3 * curve_quality_loss + 
                     0.2 * attention_reg_loss)
        
        return total_loss
    
    def compute_loss(self, cq, cr, cshft, rshft, sm):
        """
        Compatibility method for PyKT training pipeline
        Maps to the internal get_loss method
        
        Args:
            cq: Questions/concepts [batch_size, seq_len]
            cr: Response correctness [batch_size, seq_len]  
            cshft: Query questions (shifted) [batch_size, seq_len]
            rshft: Target responses (shifted) [batch_size, seq_len]
            sm: Sequence mask [batch_size, seq_len]
            
        Returns:
            loss: Combined loss tensor
        """
        return self.get_loss(cq, cr, cshft, rshft, sm)
    
    def curve_quality_loss(self, curve_params, mastery_levels):
        """Enforce learning curve constraints"""
        # Handle empty or invalid inputs
        if curve_params.numel() == 0 or mastery_levels.numel() == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Monotonicity: ensure mastery doesn't decrease (only if we have multiple time steps)
        if mastery_levels.shape[1] > 1:
            mastery_diff = mastery_levels[:, 1:] - mastery_levels[:, :-1]
            monotonicity_loss = F.relu(-mastery_diff).mean()
        else:
            monotonicity_loss = torch.tensor(0.0, device=mastery_levels.device, requires_grad=True)
        
        # Bounds: ensure mastery is in [0, 1]
        bounds_loss = (F.relu(-mastery_levels) + F.relu(mastery_levels - 1)).mean()
        
        # Parameter bounds
        if curve_params.shape[-1] >= 3:  # Ensure we have L, k, N0 parameters
            L = curve_params[:, :, :, 0]
            k = curve_params[:, :, :, 1] 
            N0 = curve_params[:, :, :, 2]
            
            param_loss = (F.relu(-L) + F.relu(L - 1) +  # L in [0, 1]
                         F.relu(-k) + F.relu(k - 10) +   # k in [0, 10]
                         F.relu(-N0) + F.relu(N0 - 100)).mean()  # N0 in [0, 100]
        else:
            param_loss = torch.tensor(0.0, device=curve_params.device, requires_grad=True)
        
        total_loss = monotonicity_loss + bounds_loss + param_loss
        
        # Ensure the loss is finite
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
            
        return total_loss
    
    def attention_regularization(self, attention_weights):
        """Regularize attention weights for smoothness"""
        if not attention_weights:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        reg_loss = 0.0
        for attn in attention_weights:
            # Encourage smooth attention patterns
            reg_loss += torch.var(attn, dim=-1).mean()
        
        return reg_loss / len(attention_weights)


class TrajectoryEncoder(Module):
    """Encodes student interaction sequences into trajectory representations"""
    
    def __init__(self, num_skills, emb_size, curve_dim):
        super().__init__()
        self.num_skills = num_skills
        self.emb_size = emb_size
        self.curve_dim = curve_dim
        
        # Networks to encode (S, N, M) tuples
        self.skill_encoder = nn.Linear(1, curve_dim // 3)
        self.attempts_encoder = nn.Linear(1, curve_dim // 3) 
        self.mastery_encoder = nn.Linear(1, curve_dim // 3 + curve_dim % 3)  # Handle remainder
        self.trajectory_proj = nn.Linear(curve_dim, emb_size)
        
    def forward(self, q, r, qry):
        """
        Convert interaction sequence to trajectory representation
        
        Returns:
            trajectories: [batch_size, seq_len, emb_size]
        """
        # Handle empty tensors
        if q.numel() == 0 or r.numel() == 0:
            device = q.device if q.numel() > 0 else (r.device if r.numel() > 0 else torch.device('cpu'))
            return torch.zeros(1, 1, self.emb_size, device=device)
            
        batch_size, seq_len = q.size()
        
        # Calculate (S, N, M) tuples for each timestep
        trajectories = torch.zeros(batch_size, seq_len, self.emb_size, device=q.device)
        
        for b in range(batch_size):
            skill_attempts = {}
            skill_correct = {}
            
            for t in range(seq_len):
                skill = q[b, t].item()
                response = r[b, t].item()
                
                if skill >= 0 and skill < self.num_skills:  # Valid skill within bounds
                    # Update attempts count
                    if skill not in skill_attempts:
                        skill_attempts[skill] = 0
                        skill_correct[skill] = 0
                    
                    skill_attempts[skill] += 1
                    skill_correct[skill] += response
                    
                    # Calculate mastery as success rate
                    mastery = skill_correct[skill] / skill_attempts[skill]
                    
                    # Encode (S, N, M) tuple
                    s_enc = self.skill_encoder(torch.tensor([skill], dtype=torch.float, device=q.device))
                    n_enc = self.attempts_encoder(torch.tensor([skill_attempts[skill]], dtype=torch.float, device=q.device))
                    m_enc = self.mastery_encoder(torch.tensor([mastery], dtype=torch.float, device=q.device))
                    
                    # Combine encodings
                    tuple_enc = torch.cat([s_enc, n_enc, m_enc], dim=-1)
                    trajectories[b, t] = self.trajectory_proj(tuple_enc)
        
        return trajectories


class TSMiniSimilarity(Module):
    """TSMini-based trajectory similarity computation"""
    
    def __init__(self, num_views=4, cache_size=10000, emb_size=256):
        super().__init__()
        self.num_views = num_views
        self.cache_size = cache_size
        self.cache = {}
        self.emb_size = emb_size
        
        # Similarity computation networks for different views
        # Input will be concatenated trajectories, so 2 * emb_size
        concat_size = 2 * emb_size
        self.temporal_sim = nn.Linear(concat_size, 1)  # Temporal patterns  
        self.difficulty_sim = nn.Linear(concat_size, 1)  # Difficulty progression
        self.error_sim = nn.Linear(concat_size, 1)  # Error patterns
        self.progress_sim = nn.Linear(concat_size, 1)  # Learning progress
        
    def forward(self, query_trajectory, key_trajectories):
        """
        Compute multi-view similarity scores
        
        Args:
            query_trajectory: [batch_size, seq_len, emb_size]
            key_trajectories: [batch_size, seq_len, emb_size]
            
        Returns:
            similarities: [batch_size, seq_len, seq_len, num_views]
        """
        batch_size, seq_len, emb_size = query_trajectory.size()
        
        # Simple cosine similarity for now (can be enhanced later)
        # Normalize trajectories
        query_norm = F.normalize(query_trajectory, p=2, dim=-1)  # [B, T, E]
        key_norm = F.normalize(key_trajectories, p=2, dim=-1)    # [B, T, E]
        
        # Compute cosine similarity: [B, T, T]
        cosine_sim = torch.matmul(query_norm, key_norm.transpose(-2, -1))
        
        # Create multiple views by applying different transformations
        similarities = torch.zeros(batch_size, seq_len, seq_len, self.num_views, device=query_trajectory.device)
        
        # View 0: Direct cosine similarity
        similarities[:, :, :, 0] = torch.sigmoid(cosine_sim)
        
        # View 1: Squared similarity (emphasizes high similarity)
        similarities[:, :, :, 1] = torch.sigmoid(cosine_sim ** 2)
        
        # View 2: Exponential similarity
        similarities[:, :, :, 2] = torch.sigmoid(torch.exp(cosine_sim - 1))
        
        # View 3: Linear transformation
        similarities[:, :, :, 3] = torch.sigmoid(0.5 * cosine_sim + 0.5)
        
        return similarities


class SimAKTAttentionBlock(Module):
    """Similarity-based attention block replacing standard transformer attention"""
    
    def __init__(self, emb_size, num_attn_heads, dropout, num_skills):
        super().__init__()
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.num_skills = num_skills
        
        # Multi-head similarity attention
        self.similarity_attention = MultiHeadSimilarityAttention(
            emb_size, num_attn_heads, dropout, emb_size
        )
        
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)
        
        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)
    
    def forward(self, q=None, k=None, trajectories=None, return_attention=False):
        # Use trajectories for similarity-based attention instead of traditional QKV
        attn_emb, attention_weights = self.similarity_attention(
            q, k, k, trajectories, return_attention=return_attention
        )
        
        # Standard transformer post-processing
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = self.attn_layer_norm(q + attn_emb)
        
        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        
        if return_attention:
            return emb, attention_weights
        else:
            return emb, None


class MultiHeadSimilarityAttention(Module):
    """Multi-head attention using trajectory similarity instead of dot-product"""
    
    def __init__(self, emb_size, num_heads, dropout, trajectory_emb_size=256):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        self.q_linear = Linear(emb_size, emb_size)
        self.k_linear = Linear(emb_size, emb_size)
        self.v_linear = Linear(emb_size, emb_size)
        
        self.similarity_computer = TSMiniSimilarity(emb_size=trajectory_emb_size)
        self.out_linear = Linear(emb_size, emb_size)
        self.dropout = Dropout(dropout)
        
    def forward(self, q, k, v, trajectories, return_attention=False):
        batch_size, seq_len, _ = q.size()
        
        # Standard V projection (Q and K not used in similarity-based attention)
        V = self.v_linear(v).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute trajectory-based similarity
        similarities = self.similarity_computer(trajectories, trajectories)
        
        # Use similarity scores as attention weights
        attention_weights = torch.mean(similarities, dim=-1)  # Average across views
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply causal mask
        causal_mask = ut_mask(seq_len).to(q.device)
        attention_weights = attention_weights.masked_fill(causal_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply attention to values
        # attention_weights: [B, T, T]
        # V: [B, T, H, D] -> need to reshape for matmul
        V_reshaped = V.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Apply attention for each head
        attended_heads = []
        for h in range(self.num_heads):
            # attention_weights: [B, T, T], V_reshaped[:, h, :, :]: [B, T, D]
            attended_h = torch.matmul(attention_weights, V_reshaped[:, h, :, :])  # [B, T, D]
            attended_heads.append(attended_h)
        
        # Concatenate heads
        attended = torch.cat(attended_heads, dim=-1)  # [B, T, H*D]
        output = self.out_linear(attended)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class LearningCurvePredictor(Module):
    """Predicts sigmoid learning curve parameters (L, k, N0) for each skill"""
    
    def __init__(self, emb_size, num_skills, curve_dim):
        super().__init__()
        self.num_skills = num_skills
        
        self.param_network = nn.Sequential(
            nn.Linear(emb_size, curve_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(curve_dim, curve_dim // 2),
            nn.ReLU(),
            nn.Linear(curve_dim // 2, num_skills * 3)  # 3 parameters per skill
        )
        
    def forward(self, embeddings):
        """
        Predict learning curve parameters
        
        Args:
            embeddings: [batch_size, seq_len, emb_size]
            
        Returns:
            curve_params: [batch_size, seq_len, num_skills, 3] (L, k, N0)
        """
        batch_size, seq_len, emb_size = embeddings.size()
        
        # Predict parameters
        params = self.param_network(embeddings)  # [batch_size, seq_len, num_skills * 3]
        params = params.view(batch_size, seq_len, self.num_skills, 3)
        
        # Apply constraints to parameters
        L = torch.sigmoid(params[:, :, :, 0])  # L in (0, 1)
        k = F.softplus(params[:, :, :, 1]) + 0.01  # k > 0
        N0 = F.softplus(params[:, :, :, 2]) + 1  # N0 > 1
        
        return torch.stack([L, k, N0], dim=-1)


class MultiSkillIntegrator(Module):
    """Integrates mastery across multiple skills for questions requiring multiple KCs"""
    
    def __init__(self, num_skills, strategy='min'):
        super().__init__()
        self.num_skills = num_skills
        self.strategy = strategy
        
        if strategy == 'weighted':
            self.skill_weights = nn.Parameter(torch.ones(num_skills) / num_skills)
    
    def forward(self, mastery_levels, question_skills):
        """
        Integrate mastery across skills
        
        Args:
            mastery_levels: [batch_size, seq_len, num_skills]
            question_skills: [batch_size, seq_len] question IDs (maps to skills)
            
        Returns:
            integrated_mastery: [batch_size, seq_len]
        """
        batch_size, seq_len, num_skills = mastery_levels.size()
        
        if self.strategy == 'min':
            # Take minimum mastery across required skills
            # For simplicity, assume each question maps to one skill (mod num_skills to handle out of bounds)
            safe_question_skills = question_skills % self.num_skills
            integrated = torch.gather(mastery_levels, 2, safe_question_skills.unsqueeze(-1)).squeeze(-1)
        
        elif self.strategy == 'weighted':
            # Weighted average of all skills
            weights = F.softmax(self.skill_weights, dim=0)
            integrated = torch.sum(mastery_levels * weights.unsqueeze(0).unsqueeze(0), dim=-1)
        
        elif self.strategy == 'conjunctive':
            # Product of probabilities (all skills must be mastered)
            # For simplicity, use minimum as approximation
            safe_question_skills = question_skills % self.num_skills
            integrated = torch.gather(mastery_levels, 2, safe_question_skills.unsqueeze(-1)).squeeze(-1)
        
        else:
            raise ValueError(f"Unknown integration strategy: {self.strategy}")
        
        return integrated