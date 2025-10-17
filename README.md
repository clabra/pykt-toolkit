<!-- Normalized README without outer fenced block -->
# pyKT

[![Downloads](https://pepy.tech/badge/pykt-toolkit)](https://pepy.tech/project/pykt-toolkit)
[![GitHub Issues](https://img.shields.io/github/issues/pykt-team/pykt-toolkit.svg)](https://github.com/pykt-team/pykt-toolkit/issues)
[![Documentation](https://img.shields.io/website/http/pykt-team.github.io/index.html?down_color=red&down_message=offline&up_message=online)](https://pykt.org/)

pyKT is a Python library built upon PyTorch to train deep learning based knowledge tracing models. The library provides a standardized set of integrated preprocessing procedures on more than seven popular datasets across different domains, five detailed prediction scenarios, and more than ten frequently compared DLKT approaches for transparent and extensive experiments. More details are available at our [website](https://pykt.org/) and [documentation](https://pykt-toolkit.readthedocs.io/en/latest/quick_start.html).

## Installation

We recommend using a dedicated conda environment followed by installation from PyPI:


```bash
conda create --name=pykt python=3.7.5
source activate pykt  # or: conda activate pykt
pip install -U pykt-toolkit -i https://pypi.python.org/simple
```

**Ubuntu/Linux note:** If you encounter `ModuleNotFoundError: No module named 'tkinter'`, install the system Tk library:

```bash
sudo apt update && sudo apt install python3-tk
```

## Hyper parameter tuning results

Hyper parameter tuning results for all DLKT models across datasets:
[Google Drive folder](https://drive.google.com/drive/folders/1MWYXj73Ke3zC6bm3enu1gxQQKAHb37hz?usp=drive_link)

## References

### Projects

1. [knowledge-tracing-collection-pytorch](https://github.com/hcnoh/knowledge-tracing-collection-pytorch)
2. [SAKT-pytorch](https://github.com/arshadshk/SAKT-pytorch)
3. [SAKT](https://github.com/shalini1194/SAKT)
4. [SAINT-pytorch](https://github.com/arshadshk/SAINT-pytorch)
5. [SAINT_plus-Knowledge-Tracing-](https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-)
6. [AKT](https://github.com/arghosh/AKT)
7. [Knowledge-Query-Network-for-Knowledge-Tracing](https://github.com/JSLBen/Knowledge-Query-Network-for-Knowledge-Tracing)
8. [ATKT](https://github.com/xiaopengguo/ATKT)
9. [GKT](https://github.com/jhljx/GKT)
10. [HawkesKT](https://github.com/THUwangcy/HawkesKT)
11. [iekt](https://github.com/ApexEDM/iekt)
12. [SKVMN/model.py](https://github.com/Badstu/CAKT_othermodels/blob/0c28d870c0d5cf52cc2da79225e372be47b5ea83/SKVMN/model.py)
13. [EduKTM](https://github.com/bigdata-ustc/EduKTM)
14. [RKT](https://github.com/shalini1194/RKT)
15. [DIMKT](https://github.com/shshen-closer/DIMKT)
16. [FoLiBi](https://github.com/skewondr/FoLiBi)
17. [DTransformer](https://github.com/yxonic/DTransformer)
18. [ReKT](https://github.com/lilstrawberry/ReKT)

### Papers

1. DKT: Deep knowledge tracing
2. DKT+: Addressing two problems in deep knowledge tracing via prediction-consistent regularization
3. DKT-Forget: Augmenting knowledge tracing by considering forgetting behavior
4. KQN: Knowledge query network for knowledge tracing: How knowledge interacts with skills
5. DKVMN: Dynamic key-value memory networks for knowledge tracing
6. ATKT: Enhancing Knowledge Tracing via Adversarial Training
7. GKT: Graph-based knowledge tracing: modeling student proficiency using graph neural network
8. SAKT: A self-attentive model for knowledge tracing
9. SAINT: Towards an appropriate query, key, and value computation for knowledge tracing
10. AKT: Context-aware attentive knowledge tracing
11. HawkesKT: Temporal Cross-Effects in Knowledge Tracing
12. IEKT: Tracing Knowledge State with Individual Cognition and Acquisition Estimation
13. SKVMN: Knowledge Tracing with Sequential Key-Value Memory Networks
14. LPKT: Learning Process-consistent Knowledge Tracing
15. QIKT: Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations
16. RKT: Relation-aware Self-attention for Knowledge Tracing
17. DIMKT: Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect
18. ATDKT: Enhancing Deep Knowledge Tracing with Auxiliary Tasks
19. simpleKT: A Simple but Tough-to-beat Baseline for Knowledge Tracing
20. SparseKT: Towards Robust Knowledge Tracing Models via K-sparse Attention
21. FoLiBiKT: Forgetting-aware Linear Bias for Attentive Knowledge Tracing
22. DTransformer: Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer
23. stableKT: Enhancing Length Generalization for Attention Based Knowledge Tracing Models with Linear Biases
24. extraKT: Extending Context Window of Attention Based Knowledge Tracing Models via Length Extrapolation
25. csKT: Addressing Cold-start Problem in Knowledge Tracing via Kernel Bias and Cone Attention
26. LefoKT: Rethinking and Improving Student Learning and Forgetting Processes for Attention Based Knowledge Tracing Models
27. FlucKT: Cognitive Fluctuations Enhanced Attention Network for Knowledge Tracing
28. UKT: Uncertainty-aware Knowledge Tracing
29. HCGKT: Hierarchical Contrastive Graph Knowledge Tracing with Multi-level Feature Learning
30. RobustKT: Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data

## Citation

We now have a [paper](https://arxiv.org/abs/2206.11460?context=cs.CY) to cite for the pyKT library:

```bibtex
@inproceedings{liupykt2022,
  title={pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models},
  author={Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and Huang, Shuyan and Tang, Jiliang and Luo, Weiqi},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

## Testing

We include lightweight, dependency-free unit tests to validate interpretability constraint logic introduced for experimental models (e.g., `GainAKT2Exp`).

### Available Tests

Current tests focus on the interpretability loss components:

- All constraint weights set to zero produce a zero interpretability loss.
- Non-zero weights yield a positive auxiliary loss tensor.
- The newly added consistency loss term contributes only when its weight is non-zero.
- Direct invocation of `compute_interpretability_loss` returns 0.0 with all weights disabled and a positive tensor when any weight is enabled.

### Lightweight Test Runner

A custom runner (`tests/run_unit_tests.py`) discovers and executes any `test_*.py` file in `tests/` without requiring `pytest`.

Run all tests:

```bash
python tests/run_unit_tests.py
```

Expected output example:

```text
Test Results Summary
=====================
PASS  test_interpretability_loss.py:test_interpretability_loss_all_zero_weights
PASS  test_interpretability_loss.py:test_interpretability_loss_nonzero_components
PASS  test_interpretability_loss.py:test_consistency_loss_effect
PASS  test_interpretability_loss.py:test_compute_interpretability_loss_direct

Totals: PASS=4 FAIL=0
All tests passed.
```

### Adding New Tests

1. Create `tests/test_<topic>.py`.
2. Define functions whose names start with `test_`.
3. Re-run the test runner.

### Continuous Integration (Example Snippet)

```bash
python -m pip install -e .
python tests/run_unit_tests.py
```

### Notes

- Interpretability tests seed Torch for deterministic behavior.
- No external frameworks are required; `pytest` can be integrated later if broader coverage is needed.
- For performance-sensitive environments large tests can be temporarily relocated.

## Analysis Artifacts Index

An index of ablation and interpretability analysis documents is maintained in `paper/INDEX.md`.

Quick link:

- [Analysis Documents Index](paper/INDEX.md)

Use the redo ablation report referenced there for any claims regarding constraint impact (the original report includes an erratum explaining a baseline misconfiguration).
