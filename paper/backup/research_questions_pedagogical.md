================================================================================
RESEARCH QUESTIONS & HYPOTHESES: iKT2 Interpretability
PEDAGOGICAL FRAMING (Alternative to Psychometric Framing)
================================================================================

## From Claim to Questions: Pedagogical Perspective

### THE CLAIM (Answer - Reframed):
"We assess iKT2's interpretability through three complementary metrics grounded 
in learning theory. Prediction consistency (r=0.76, p<1e-272) demonstrates that 
the model's internal understanding of student ability and task difficulty produces 
mastery estimates that align with actual performance predictions, validating that 
the learned representations capture meaningful educational constructs. Task 
difficulty coherence (r=0.57, p<1e-8) confirms that learned difficulty rankings 
align with empirically observed challenge levels. Learning progression validity 
(time-lagged r=0.57, p<1e-75) validates that mastery trajectories capture skill 
acquisition patterns comparably to established learning models, demonstrating 
that data-driven representations preserve pedagogical interpretability."

================================================================================
## THE QUESTIONS (What the claim answers):
================================================================================

### PRIMARY RESEARCH QUESTION (Overarching):

**RQ0: Can a deep learning model for student assessment be both accurate and 
pedagogically interpretable by grounding representations in learning principles?**

Sub-question: Does incorporating explicit ability-difficulty interactions and 
skill mastery tracking into a neural architecture maintain interpretability of 
classical learning models while achieving competitive prediction accuracy?

---

### SPECIFIC RESEARCH QUESTIONS:

#### 1. PREDICTION CONSISTENCY (Internal Coherence)

**RQ1: Does the model's internal understanding of learning states produce 
predictions consistent with observed performance?**

**Pedagogical Foundation:**
Learning theory posits that performance depends on the interaction between 
learner ability and task difficulty. A pedagogically interpretable model should 
make consistent predictions whether reasoning through explicit skill mastery 
estimates or direct performance prediction.

**Operationalized as:**
- Q1.1: Is there significant correlation between skill mastery estimates (M) 
  and performance predictions (p_correct)?
- Q1.2: Does this correlation demonstrate that learned ability (θ) and difficulty 
  (β) representations are educationally meaningful rather than arbitrary features?
- Q1.3: Does the ability-difficulty interaction M = σ(θ - β) produce valid 
  mastery probabilities aligned with learning theory?

**Hypothesis H1:**
The model's two prediction mechanisms (direct performance and ability-difficulty 
mastery) will produce highly correlated outputs (r > 0.7), demonstrating that 
internal representations capture meaningful learning constructs rather than 
opaque neural features.

**Null Hypothesis H0₁:**
The correlation between mastery estimates and performance predictions is weak 
(r < 0.5) or non-significant, indicating that internal representations are 
disconnected from the actual prediction mechanism and lack pedagogical meaning.

**Metric:** Prediction Consistency = Pearson(M_mastery, p_correct)
**Result:** r = 0.76, p < 1e-272 → **H0₁ rejected, H1 supported**

**Educational Interpretation:**
- Model reasons about learning in interpretable terms (ability vs difficulty)
- Parameters θ and β represent meaningful educational constructs
- Prediction mechanism is transparent and pedagogically grounded

**Why this matters for educators:**
Teachers can understand *why* the model predicts a student will struggle: either 
the student's current ability is insufficient, or the task is too difficult. This 
enables targeted interventions (skill remediation vs task scaffolding).

---

#### 2. TASK DIFFICULTY COHERENCE (Empirical Grounding)

**RQ2: Do the model's learned difficulty representations align with empirically 
observed task challenge levels?**

**Pedagogical Foundation:**
In educational settings, task difficulty is not arbitrary but reflects objective 
challenge levels observed across many students. A pedagogically valid model should 
learn difficulty rankings that match empirical evidence (e.g., tasks that most 
students fail should be recognized as harder).

**Operationalized as:**
- Q2.1: Is there significant rank-order correlation between learned difficulties 
  (β_learned) and empirically observed difficulties (β_empirical)?
- Q2.2: Does training preserve empirical difficulty orderings?
- Q2.3: Can we recover meaningful difficulty rankings from the trained model for 
  curriculum design?

**Hypothesis H2:**
Learned difficulty representations will maintain moderate-to-strong correlation 
(r > 0.5) with empirically observed challenge levels, demonstrating that the 
model respects real-world educational evidence while adapting to local patterns.

**Null Hypothesis H0₂:**
Learned difficulties show weak or no correlation (r < 0.3) with observed 
challenge levels, indicating that training produces arbitrary difficulty 
estimates disconnected from educational reality.

**Metric:** Task Difficulty Coherence = Pearson(β_learned, β_empirical)
**Result:** r = 0.57, p < 1e-8 → **H0₂ rejected, H2 supported**

**Educational Interpretation:**
- Model learns from data but respects empirical evidence
- Difficulty estimates are not black-box features but pedagogically meaningful
- Rankings can inform curriculum sequencing and adaptive task selection

**Why this matters for educators:**
Curriculum designers can trust that the model's difficulty estimates reflect 
actual student experience. Tasks the model identifies as "hard" are genuinely 
challenging, not artifacts of the neural architecture.

---

#### 3. LEARNING PROGRESSION VALIDITY (Developmental Alignment)

**RQ3: Does the model capture skill acquisition patterns consistent with 
established learning theories?**

**Pedagogical Foundation:**
Learning is a progressive process: students move from low mastery to high mastery 
through practice and instruction. Models grounded in learning theory should produce 
mastery trajectories that align with established frameworks for skill acquisition 
(e.g., knowledge construction, spaced repetition effects, mastery learning).

**Operationalized as:**
- Q3.1: Do mastery trajectories from iKT2 correlate with trajectories from 
  established learning models (e.g., knowledge tracing frameworks)?
- Q3.2: Does data-driven learning of initial states compromise the ability to 
  track pedagogically valid progression?
- Q3.3: Are learning curves interpretable through classical learning theory 
  (practice effects, forgetting, transfer)?

**Hypothesis H3:**
Despite using data-driven initialization instead of expert-specified priors, 
iKT2 mastery trajectories will show moderate-to-strong correlation (r > 0.5) 
with established learning models when accounting for initialization differences, 
demonstrating that learned representations capture genuine skill acquisition 
patterns.

**Null Hypothesis H0₃:**
Mastery trajectories show weak correlation (r < 0.3) with established learning 
models, indicating that deep learning dynamics are fundamentally different from 
pedagogical theories of skill acquisition and therefore not interpretable through 
learning frameworks.

**Metric:** Learning Progression Validity (time-lagged) = Pearson(M_iKT2, M_baseline) 
for attempts > 3
**Result:** r = 0.57, p < 1e-75 → **H0₃ rejected, H3 supported**

**Educational Interpretation:**
- Model captures real learning dynamics (not just pattern matching)
- Data-driven approach doesn't sacrifice pedagogical validity
- Trajectories reflect meaningful skill development over time

**Why this matters for educators:**
Learning analytics can be trusted to reflect actual skill development. When the 
model shows a student's mastery increasing, it reflects genuine learning (practice 
effects, knowledge consolidation) rather than arbitrary neural dynamics.

---

================================================================================
## HIERARCHICAL STRUCTURE OF QUESTIONS:
================================================================================

```
RQ0: Can DL be pedagogically interpretable?
  │
  ├─ RQ1: Prediction Consistency
  │   ├─ Q1.1: Mastery ≈ Performance?
  │   ├─ Q1.2: Are ability/difficulty meaningful?
  │   └─ Q1.3: Does interaction model work?
  │   → Validates: Transparent reasoning about learning
  │
  ├─ RQ2: Task Difficulty Coherence  
  │   ├─ Q2.1: β_learned ≈ β_empirical?
  │   ├─ Q2.2: Training preserves evidence?
  │   └─ Q2.3: Can inform curriculum design?
  │   → Validates: Empirical grounding maintained
  │
  └─ RQ3: Learning Progression Validity
      ├─ Q3.1: iKT2 ≈ established learning models?
      ├─ Q3.2: Data-driven okay for pedagogy?
      └─ Q3.3: Curves interpretable via learning theory?
      → Validates: Captures genuine skill acquisition
```

================================================================================
## FORMAL HYPOTHESES SUMMARY:
================================================================================

### Main Hypothesis (H_main):
A neural architecture with explicit ability-difficulty interactions and skill 
mastery tracking can achieve both competitive prediction accuracy AND pedagogical 
interpretability, as evidenced by:
1. Strong internal prediction consistency (dual prediction coherence)
2. Preservation of empirical difficulty evidence (task coherence)
3. Alignment with established learning progression patterns (developmental validity)

### Alternative Hypotheses:

**H_alt1 (Performance-Interpretability Trade-off):**
High prediction accuracy requires complex non-interpretable features
 Rejected: iKT2 achieves AUC=0.71 with transparent ability-difficulty reasoning

**H_alt2 (Ability-Difficulty as Superficial Structure):**
Explicit learning constructs are merely constraints with no functional role
 Rejected: Prediction consistency r=0.76 shows constructs are functional

**H_alt3 (Deep  Learning Theory):**Learning 
Neural network learning dynamics fundamentally differ from pedagogical theories
 Rejected: Progression validity r=0.57 shows comparable dynamics

================================================================================
## CONTRIBUTION CLAIMS (Pedagogical Framing):
================================================================================

Based on these research questions, the paper contributes:

1. **Methodological Contribution:**
   - Neural architecture grounded in learning principles (ability-difficulty interaction)
   - Three-metric validation framework for pedagogical interpretability
   - Data-driven initialization that preserves educational meaningfulness

2. **Empirical Contribution:**
   - Evidence that DL can maintain pedagogical interpretability
   - Quantification of interpretability through learning-theory-aligned metrics
   - Demonstration that data-driven learning preserves educational validity

3. **Theoretical Contribution:**
   - Bridge between deep learning and learning sciences
   - Evidence that neural representations can align with educational theory
   - Framework for evaluating interpretability in educational AI systems

4. **Practical Contribution:**
   - Actionable insights for educators (ability vs difficulty interventions)
   - Trustworthy difficulty estimates for curriculum design
   - Valid learning analytics for tracking skill development

================================================================================
## PAPER STRUCTURE ALIGNMENT (Pedagogical Framing):
================================================================================

**Introduction:**
- Problem: DL models for education are black boxes (motivation for RQ0)
- Impact: Educators can't trust or act on predictions without understanding
- Gap: No systematic way to validate pedagogical interpretability claims
- Proposal: Learning-grounded architecture + validation framework

**Background:**
- Learning theory (skill acquisition, ability-difficulty, practice effects)
- Deep learning for student modeling → accurate but opaque
- Gap → need for pedagogically interpretable neural models

**Method:**
- iKT2 architecture → how we operationalize learning principles in DL
  * Ability tracking (θ): Student's evolving competence
  * Difficulty modeling (β): Task challenge levels
  * Mastery estimation: Interaction of ability and difficulty
- Two-phase training → how we maintain pedagogical grounding
- Validation metrics → how we measure interpretability (RQ1-3)

**Results:**
- RQ1 answer: Prediction consistency r=0.76 (transparent reasoning proven)
- RQ2 answer: Difficulty coherence r=0.57 (empirical grounding maintained)
- RQ3 answer: Progression validity r=0.57 (learning dynamics captured)

**Discussion:**
- RQ0 answer: Yes, DL + learning theory = interpretable + accurate
- Educational implications: 
  * Teachers can understand *why* predictions are made
  * Difficulty estimates can inform curriculum design
  * Mastery trajectories reflect genuine learning
- Limitations: Moderate difficulty coherence, single-skill approximation
- Future: Multi-skill transfer, metacognitive strategies, collaborative learning

**Conclusion:**
- Main claim validated through three complementary learning-theory metrics
- Contribution: Framework for pedagogically interpretable educational AI
- Impact: Trustworthy learning analytics that educators can understand and act on

================================================================================
## KEY TERMINOLOGY MAPPING:
================================================================================

### Psychometric → Pedagogical Translation:

| Psychometric Term | Pedagogical Equivalent | Educational Meaning |
|-------------------|------------------------|---------------------|
| IRT consistency | Prediction consistency | Transparent reasoning |
| Item difficulty (β_IRT) | Task difficulty (β_empirical) | Empirical challenge level |
| Person ability (θ) | Student ability | Current competence |
| Rasch model | Ability-difficulty interaction | Learning as function of readiness × challenge |
| Parameter fidelity | Empirical grounding | Alignment with observed evidence |
| BKT dynamics | Learning progression | Skill acquisition over time |
| Difficulty fidelity | Task difficulty coherence | Validity of challenge estimates |
| Head agreement | Prediction consistency | Internal reasoning coherence |
| Mastery probability | Skill mastery estimate | Likelihood of successful performance |

### Why This Matters:

**Psychometric framing:**
- Strengths: Rigorous statistical foundation, established measurement theory
- Weaknesses: Static assessment assumptions, test-theory jargon, limited to snapshot evaluation

**Pedagogical framing:**
- Strengths: Dynamic learning focus, accessible to educators, action-oriented
- Weaknesses: Less formal mathematical grounding, diverse theoretical traditions

**iKT2 bridges both:**
- Uses mathematical rigor of ability-difficulty models (from psychometrics)
- Frames in terms of learning progression and skill development (pedagogy)
- Validates through both statistical evidence AND educational meaningfulness

================================================================================
## ADDRESSING THE IRT "STATIC" CONCERN:
================================================================================

### The Problem with Psychometric Framing:

**Classical IRT Assumption:**
- Ability (θ) and difficulty (β) are STATIC parameters
- Each test-taker has a fixed ability; each item has a fixed difficulty
- Appropriate for: Standardized testing, certification exams, college admissions
- Problematic for: Learning environments where ability changes through practice

### iKT2's Pedagogical Resolution:

**Dynamic Learning Model:**
```
M_t = σ(θ_t - β)          # Mastery at time t
```

**Key Insight:**
We use the MATHEMATICAL STRUCTURE of ability-difficulty interaction (borrowed 
from IRT) but apply it in a DYNAMIC LEARNING CONTEXT where ability changes.

**Pedagogical Justification:**
- Student ability improves through practice and instruction (θ_t changes)
- Task difficulty is relatively stable (β is constant)
- Performance at any moment depends on current ability vs task challenge
- This aligns with: Zone of Proximal Development, Mastery Learning, Spaced Repetition

### Terminology Choice for Paper:

**Option 1: Hybrid (Recommended)**
"We ground iKT2 in the ability-difficulty interaction framework (Rasch, 1960), 
adapted for dynamic learning contexts where student ability evolves through 
practice..."

- Acknowledges IRT origins (gives credit, enables comparison)
- Clarifies extension to learning (addresses static concern)
- Maintains mathematical rigor

**Option 2: Pure Pedagogical**
"We model learning as the interaction between evolving student competence and 
stable task challenge, following mastery learning principles..."

- Avoids psychometric terminology entirely
- More accessible to education researchers
- May lose statistical rigor perception

**Option 3: Learning Sciences Hybrid**
"We adopt a competency-challenge framework where performance depends on the 
interaction between learner readiness and task demand, operationalized through 
a logistic model σ(θ - β) adapted from measurement theory..."

- Bridges learning sciences and psychometrics
- Emphasizes dynamic aspects
- Technical but accessible

================================================================================
## RECOMMENDED FRAMING FOR PAPER:
================================================================================

### Abstract/Introduction:
Use pedagogical framing (prediction consistency, task difficulty coherence, 
learning progression validity) to emphasize educational relevance.

### Method - Architecture:
Use hybrid framing: "ability-difficulty interaction" with clear statement that 
ability is dynamic (learned from sequence) while difficulty is task property.

### Method - Validation:
Use learning-theory grounding but cite IRT literature for mathematical foundation.

### Results:
Present metrics in pedagogical terms with educational interpretation.

### Discussion - Related Work:
Acknowledge IRT origins, discuss extension to dynamic learning contexts as 
contribution that addresses limitation of static psychometric models.

### Conclusion:
Emphasize pedagogical interpretability and actionable insights for educators.

================================================================================
