import torch
import torch.nn as nn
import numpy as np

class IDKTRoster:
    def __init__(self, students, skills, model, device="cpu"):
        """
        Initializes an iDKT Roster that mirrors pyBKT.models.Roster design.
        
        Args:
            students (list): List of student IDs.
            skills (list): List of skill IDs (sequential indices).
            model (nn.Module): Loaded iDKT model.
            device (str): Device to run inference on.
        """
        self.students = students
        self.skills = skills
        self.model = model
        self.device = device
        
        # Histories: {student_id: [(skill_id, correctness), ...]}
        self.histories = {int(uid): [] for uid in students}
        
        # Cache for current hidden states to avoid re-running prefix for every skill query
        self.state_cache = {int(uid): None for uid in students}

    def update_state(self, skill_id, student_id, correct):
        """Adds an interaction to the student's history."""
        uid = int(student_id)
        self.histories[uid].append((int(skill_id), int(correct)))
        # Invalidate cache
        self.state_cache[uid] = None

    def _get_student_state(self, student_id):
        """Computes the hidden state (knowledge state) for a student based on history."""
        uid = int(student_id)
        if self.state_cache[uid] is not None:
            return self.state_cache[uid]
            
        history = self.histories[uid]
        if not history:
            return None # No history yet
            
        # Prepare batch for iDKT
        # History is (q1, r1), (q2, r2), ...
        q_seq = [h[0] for h in history]
        r_seq = [h[1] for h in history]
        
        # iDKT expects [Batch, Seq]
        q_tensor = torch.tensor([q_seq]).long().to(self.device)
        r_tensor = torch.tensor([r_seq]).long().to(self.device)
        
        with torch.no_grad():
            # Run forward pass (qtest=True returns concat_q which contains knowledge state)
            # Actually, we want the state AFTER the last interaction.
            # in idkt.forward: d_output is knowledge retriever output.
            # we need to append a dummy question to peek the state after the last interaction?
            # No, AKT Knowledge Retriever output at index t is the state after observing prefix 0...t-1 
            # and peeking question t.
            # If we want the state AFTER interaction t (where we observed r_t), 
            # we can look at the encoder output or just run one more step.
            
            # For simplicity and consistency with Roster (which returns prob AFTER update):
            # We want the probability of mastery for skill S given history (q_1, r_1) ... (q_t, r_t).
            # This is equivalent to the model's prediction for S at step t+1.
            
            # We can use qtest=True to get the final knowledge state.
            # We need to provide a "query" skill to query the encoder.
            # But the hidden state yt^ in the encoder depends on the history.
            
            # Let's use the forward pass directly.
            # Pass prefix history and get the last hidden state.
            preds, _, _, _, concat_q = self.model(q_tensor, r_tensor, q_tensor, qtest=True)
            # last_state = concat_q[:, -1, :].clone() # [1, 2*d]
            
            # Wait, concat_q in idkt.py is torch.cat([d_output, q_embed_data], dim=-1)
            # d_output is the Knowledge State.
            # We want to use this d_output but with DIFFERENT query embeddings.
            
            # Let's just return the whole model output for the next step?
            # iDKT predicts P(r_{t+1}) given history 1...t and question t+1.
            # We can query all skills as "question t+1".
            
            # To do it efficiently:
            # 1. Get the encoded history (Encoded Interactions y^ from Encoder)
            # 2. For each query skill, run the Decoder (Knowledge Retriever) and Output Head.
            
            # This is complex to implement outside of idkt.py.
            # Simpler: Wrap the call.
            self.state_cache[uid] = (q_tensor, r_tensor)
            return self.state_cache[uid]

    def get_mastery_prob(self, skill_id, student_id):
        """Returns the mastery prob for a skill (proxied by iDKT prediction)."""
        uid = int(student_id)
        sid = int(skill_id)
        
        state_data = self._get_student_state(uid)
        if state_data is None:
            # Base prior (Initial Mastery)
            # Run model with one dummy step or just use init_mastery head on SID.
            q_dummy = torch.tensor([[sid]]).long().to(self.device)
            r_dummy = torch.tensor([[0]]).long().to(self.device) # Correctness doesn't matter for pure static
            with torch.no_grad():
                # We need a forward call that gives us the static projection.
                # In forward: initmastery = m(self.out_initmastery(q_embed_data).squeeze(-1))
                # where q_embed_data is the embedding of the current skill.
                _, initmastery, _, _ = self.model(q_dummy, r_dummy, q_dummy)
                return initmastery[0, 0].item()

        q_hist, r_hist = state_data
        # Append query skill to history with dummy response
        q_query = torch.cat([q_hist, torch.tensor([[sid]]).long().to(self.device)], dim=1)
        r_query = torch.cat([r_hist, torch.tensor([[0]]).long().to(self.device)], dim=1) # dummy r
        
        with torch.no_grad():
            preds, _, _, _ = self.model(q_query, r_query, q_query)
            return preds[0, -1].item() # prediction for the last (query) item

    def get_mastery_probs(self, student_id):
        """Returns mastery probs for all skills for a student (Batch Optimized)."""
        uid = int(student_id)
        
        state_data = self._get_student_state(uid)
        if state_data is None:
            # Base prior (Initial Mastery) for all skills
            q_all = torch.tensor(self.skills).long().to(self.device)
            r_all = torch.zeros_like(q_all).to(self.device)
            with torch.no_grad():
                # We need to broadcast q_all for the forward pass
                q_all_batch = q_all.unsqueeze(0) # [1, N]
                r_all_batch = r_all.unsqueeze(0) # [1, N]
                _, initmastery, _, _ = self.model(q_all_batch, r_all_batch, q_all_batch)
                # initmastery is [1, N]
                return {sid: p for sid, p in zip(self.skills, initmastery[0].cpu().numpy().tolist())}

        q_hist, r_hist = state_data # [1, T], [1, T]
        T = q_hist.shape[1]
        N = len(self.skills)
        
        # WE WANT: P(r_{t+1} | skill_i, history) for all i in skills
        # OPTIMIZED: Process all skills in a single batch
        # To avoid OOM for very large Skill sets, we could sub-batch, but for 100-200 it's fine.
        
        # 1. Get Knowledge State at the current (last) step
        with torch.no_grad():
            # Run model with qtest=True to get the final hidden state
            # yt^ is the knowledge state in AKT terminology
            # self.model.model returns d_output
            q_embed_data, qa_embed_data = self.model.base_emb(q_hist, r_hist)
            d_output = self.model.model(q_embed_data, qa_embed_data, None) # [1, T, D]
            
            last_h = d_output[:, -1:, :] # [1, 1, D]
            
            # 2. Get embeddings for ALL skills
            q_all = torch.tensor(self.skills).long().to(self.device)
            q_embed_all = self.model.q_embed(q_all) # [N, D]
            q_embed_all = q_embed_all.unsqueeze(0).unsqueeze(0) # [1, 1, N, D]
            
            # 3. Broadcast Knowledge State
            last_h_rep = last_h.unsqueeze(2).repeat(1, 1, N, 1) # [1, 1, N, D]
            
            # 4. Concatenate and predict
            concat_all = torch.cat([last_h_rep, q_embed_all], dim=-1) # [1, 1, N, 2D]
            
            # Run through output head (Sigmoid(Linear))
            m = nn.Sigmoid()
            preds = m(self.model.out(concat_all)).squeeze() # [N]
            
            return {sid: p for sid, p in zip(self.skills, preds.cpu().numpy().tolist())}

    def get_mastery_matrix(self, student_id):
        """
        Returns the FULL [Steps x Skills] mastery matrix for a student.
        Extremely efficient: O(T^2 + T*N) instead of O(N*T^3).
        """
        uid = int(student_id)
        history = self.histories[uid]
        if not history:
            return None
            
        q_seq = [h[0] for h in history]
        r_seq = [h[1] for h in history]
        T = len(q_seq)
        N = len(self.skills)
        
        q_tensor = torch.tensor([q_seq]).long().to(self.device)
        r_tensor = torch.tensor([r_seq]).long().to(self.device)
        
        with torch.no_grad():
            # 1. Encode full history
            q_embed_data, qa_embed_data = self.model.base_emb(q_tensor, r_tensor)
            h_seq = self.model.model(q_embed_data, qa_embed_data, None) # [1, T, D]
            
            # 2. Get embeddings for ALL skills
            q_all = torch.tensor(self.skills).long().to(self.device)
            q_embed_all = self.model.q_embed(q_all) # [N, D]
            
            # 3. Broadcast and query all skills at all steps
            # h_seq: [1, T, D] -> [T, 1, D]
            # q_embed_all: [N, D] -> [1, N, D]
            h_expanded = h_seq.squeeze(0).unsqueeze(1).repeat(1, N, 1) # [T, N, D]
            q_expanded = q_embed_all.unsqueeze(0).repeat(T, 1, 1) # [T, N, D]
            
            concat_matrix = torch.cat([h_expanded, q_expanded], dim=-1) # [T, N, 2D]
            
            # 4. Predict
            m = nn.Sigmoid()
            # self.model.out is Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear)
            # It works on arbitrary batch dimensions.
            matrix_preds = m(self.model.out(concat_matrix)).squeeze(-1) # [T, N]
            
            return matrix_preds.cpu().numpy()
