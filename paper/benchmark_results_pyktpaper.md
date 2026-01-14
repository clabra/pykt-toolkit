# Table 2: Overall Prediction Performance in AUC (PyKT Benchmark 2023)

Based on the results reported in Table 2 of ["PYKT - A Python Library to Benchmark Deep Learning based Knowledge Tracing Models" (Liu et al. 2023)](../../bibliography/papers-pykt/2023%20Liu%20_%20PYKT%20-%20%20A%20Python%20Library%20to%20Benchmark%20Deep%20Learning%20based%20Knowledge%20Tracing%20Models.pdf).

## Question Level (All-in-One) - AUC

| Model | AS2009 | AL2005 | BD2006 | NIPS34 | Statics2011 | AS2015 | POJ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DKT | 0.7541 | 0.8149 | 0.8015 | 0.7689 | 0.8222 | 0.7271 | 0.6089 |
| DKT+ | 0.7547 | 0.8156 | 0.8020 | 0.7696 | 0.8279 | 0.7285 | 0.6173 |
| DKT-F | - | 0.8147 | 0.7985 | 0.7733 | 0.7839 | 0.7254 | 0.6030 |
| KQN | 0.7477 | 0.8027 | 0.7936 | 0.7684 | 0.8232 | 0.7227 | 0.6080 |
| DKVMN | 0.7473 | 0.8054 | 0.7983 | 0.7673 | 0.8093 | 0.7245 | 0.6056 |
| ATKT | 0.7470 | 0.7995 | 0.7889 | 0.7665 | 0.8055 | 0.7258 | 0.6075 |
| GKT | 0.7424 | 0.8110 | 0.8046 | 0.7689 | 0.8040 | 0.7114 | 0.6070 |
| SAKT | 0.7246 | 0.7880 | 0.7740 | 0.7517 | 0.7965 | 0.7026 | 0.6095 |
| SAINT | 0.6958 | 0.7775 | 0.7781 | 0.7873 | 0.7599 | - | 0.5563 |
| **AKT** | **0.7853** | **0.8306** | **0.8208** | **0.8033** | **0.8309** | **0.7281** | **0.6281** |

## KC Level (All-in-One) - AUC

| Model | AS2009 | AL2005 | BD2006 | NIPS34 |
| :--- | :---: | :---: | :---: | :---: |
| DKT | 0.7419 | 0.8146 | 0.8013 | 0.7681 |
| DKT+ | 0.7424 | 0.8144 | 0.8019 | 0.7689 |
| DKT-F | - | 0.8163 | 0.7984 | 0.7727 |
| KQN | 0.7361 | 0.8005 | 0.7935 | 0.7677 |
| DKVMN | 0.7330 | 0.7891 | 0.7981 | 0.7668 |
| ATKT | 0.7337 | 0.7964 | 0.7885 | 0.7658 |
| GKT | 0.7227 | 0.8025 | 0.8045 | 0.7681 |
| SAKT | 0.7085 | 0.7682 | 0.7738 | 0.7516 |
| SAINT | 0.6865 | 0.6662 | 0.7779 | 0.7860 |
| **AKT** | **0.7650** | **0.8091** | **0.8206** | **0.8017** |

> **Observation 2.** One-by-one evaluation on expanded KC sequences causes label leakage problem that leads to performance inflation.
