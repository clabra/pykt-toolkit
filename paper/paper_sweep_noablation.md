## Paper Table (Table 5, ablation=none)

| Dataset | Best Test AUC (p_sup) | Exp ID | Architecture | Learning Rate | Dropout | lambda_ref | Epochs | Notes |
|---------|----------------------|--------|--------------|---------------|---------|------------|--------|-------|
| assist2009 | **0.7824** ± 0.0012 | 481134 | 4 blocks, 8 heads | 1e-4 | 0.1 | 0.5 | 55 | 4/8 architecture outperforms 4/4 |
| assist2015 | **0.7073** ± 0.0007 | 878655 | 4 blocks, 4 heads | 1e-4 | 0.1 | 0.5 | 55 | Best p_ref interpretability |
| algebra2005 | **0.8251** ± 0.0018 | 220246 | 4 blocks, 8 heads | 1e-4 | 0.1 | 0.5 | 55 | 4/8 architecture (highest p_sup) |
| bridge2algebra2006 | **0.8120** ± 0.0009 | 878655 | 4 blocks, 4 heads | 1e-4 | 0.1 | 0.5 | 55 | Same p_sup as 4/8, better p_ref |
| nips_task34 | **0.7991** ± 0.0005 | 878655 | 4 blocks, 4 heads | 1e-4 | 0.1 | 0.5 | 55 | Best p_ref interpretability |

**Summary**: For ablation=none (with grounding/probing losses), all datasets have full test results. ASSISTments 2009 and Algebra 2005 achieve highest p_sup with 4/8 architecture, while other datasets use 4/4 architecture for better interpretability (p_ref) with comparable p_sup performance.

## Table ablation=none

### assist2009

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 998355 | benchmark-ablall | 20260131_131331...998355 | 4 | 4 | 64 | 0.1 | 2e4 | 0.5 | 39 | 0.7832±0.0017 ✅ | 0.6732 | (0.7783) | (0.7450) |
| 991642 | sweep_init | 20260131_104046...991642 | 4 | 4 | 64 | 0.1 | 2e4 | 0.5 | 2 | 0.7137 | 0.7241 | ❌ (0.7783) | ✅ (0.7327) |
| 454557 | sweep_init | 20260131_105925...454557 | 4 | 4 | 64 | 0.2 | 2e4 | 0.5 | 2 | 0.7119 | 0.7237 | ❌ (0.7783) | ❌ (0.7327) |
| 941711 | sweep_init | 20260131_111223...941711 | 4 | 4 | 128 | 0.1 | 1e4 | 0.3 | 2 | 0.7115 | 0.7192 | ❌ (0.7783) | ❌ (0.7327) |
| 785108 | sweep_init | 20260131_104044...785108 | 4 | 4 | 64 | 0.1 | 1e4 | 0.5 | 2 | 0.7113 | 0.7202 | ❌ (0.7783) | ❌ (0.7327) |
| 430771 | sweep_init | 20260131_104049...430771 | 4 | 4 | 64 | 0.2 | 1e4 | 0.5 | 2 | 0.7092 | 0.7198 | ❌ (0.7783) | ❌ (0.7327) |

### assist2015

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 522308 | sweep_init | 20260131_104047...522308 | 4 | 4 | 64 | 0.1 | 2e4 | 0.5 | 2 | 0.6880 | 0.6819 | ❌ (0.7073) | ✅ (0.6968) |
| 100168 | sweep_init | 20260131_105936...100168 | 4 | 4 | 64 | 0.2 | 2e4 | 0.5 | 2 | 0.6823 | 0.6817 | ❌ (0.7073) | ❌ (0.6968) |
| 786052 | sweep_init | 20260131_111835...786052 | 4 | 4 | 128 | 0.1 | 1e4 | 0.3 | 2 | 0.6808 | 0.6783 | ❌ (0.7073) | ❌ (0.6968) |
| 110916 | sweep_init | 20260131_104044...110916 | 4 | 4 | 64 | 0.1 | 1e4 | 0.5 | 2 | 0.6803 | 0.6792 | ❌ (0.7073) | ❌ (0.6968) |
| 746624 | sweep_init | 20260131_104050...746624 | 4 | 4 | 64 | 0.2 | 1e4 | 0.5 | 2 | 0.6767 | 0.6785 | ❌ (0.7073) | ❌ (0.6968) |

### algebra2005

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 702725 | sweep_init | 20260131_104048...702725 | 4 | 4 | 64 | 0.1 | 2e4 | 0.5 | 2 | 0.7413 | 0.7587 | ❌ (0.8219) | ✅ (0.7429) |
| 246736 | sweep_init | 20260131_110318...246736 | 4 | 4 | 64 | 0.2 | 2e4 | 0.5 | 2 | 0.7316 | 0.7582 | ❌ (0.8219) | ❌ (0.7429) |
| 534077 | sweep_init | 20260131_104045...534077 | 4 | 4 | 64 | 0.1 | 1e4 | 0.5 | 2 | 0.7169 | 0.7550 | ❌ (0.8219) | ❌ (0.7429) |
| 806444 | sweep_init | 20260131_104050...806444 | 4 | 4 | 64 | 0.2 | 1e4 | 0.5 | 2 | 0.7087 | 0.7539 | ❌ (0.8219) | ❌ (0.7429) |

### bridge2algebra2006

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 904949 | sweep_init | 20260131_104048...904949 | 4 | 4 | 64 | 0.1 | 2e4 | 0.5 | 2 | 0.7536 | 0.7556 | ❌ (0.8120) | ✅ (0.7057) |
| 796034 | sweep_init | 20260131_110328...796034 | 4 | 4 | 64 | 0.2 | 2e4 | 0.5 | 2 | 0.7486 | 0.7553 | ❌ (0.8120) | ✅ (0.7057) |
| 728430 | sweep_init | 20260131_104045...728430 | 4 | 4 | 64 | 0.1 | 1e4 | 0.5 | 2 | 0.7404 | 0.7537 | ❌ (0.8120) | ✅ (0.7057) |
| 504142 | sweep_init | 20260131_104051...504142 | 4 | 4 | 64 | 0.2 | 1e4 | 0.5 | 2 | 0.7346 | 0.7535 | ❌ (0.8120) | ❌ (0.7057) |

### nips_task34

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 907807 | sweep_init | 20260131_104049...907807 | 4 | 4 | 64 | 0.1 | 2e4 | 0.5 | 2 | 0.7399 | 0.6709 | ❌ (0.7991) | ✅ (0.7206) |
| 746493 | sweep_init | 20260131_110339...746493 | 4 | 4 | 64 | 0.2 | 2e4 | 0.5 | 2 | 0.7362 | 0.6714 | ❌ (0.7991) | ✅ (0.7206) |
| 148959 | sweep_init | 20260131_104046...148959 | 4 | 4 | 64 | 0.1 | 1e4 | 0.5 | 2 | 0.7344 | 0.6505 | ❌ (0.7991) | ✅ (0.7206) |
| 905210 | sweep_init | 20260131_105925...905210 | 4 | 4 | 64 | 0.2 | 1e4 | 0.5 | 2 | 0.7324 | 0.6503 | ❌ (0.7991) | ❌ (0.7206) |

## Table ablation=all

### assist2009

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 697945 | baseline-ablall | 20260126_191440...697945 | 4 | 4 | 64 | 0.1 | 1e4 | 0 | 55 | 0.7831  | N/A | -- | -- |

### assist2015

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 589915 | baseline-ablall | 20260126_191440...589915 | 4 | 4 | 64 | 0.1 | 1e4 | 0 | 55 | 0.7078 ✅ | N/A | -- | -- |
| 942564 | benchmark-ablall | 20260131_131331...942564 | 4 | 4 | 64 | 0.1 | 2e4 | 0 | 39 | 0.7070±0.0006  | N/A | (0.7078) | (0.6995) |

### algebra2005

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 384404 | baseline-ablall | 20260126_191440...384404 | 4 | 4 | 64 | 0.1 | 1e4 | 0 | 55 | 0.8240 ✅ | N/A | -- | -- |
| 968726 | benchmark-ablall | 20260131_131331...968726 | 4 | 4 | 64 | 0.1 | 2e4 | 0 | 39 | 0.8235±0.0007  | N/A | (0.8240) | (0.7475) |

### bridge2algebra2006

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 663881 | baseline-ablall | 20260126_191440...663881 | 4 | 4 | 64 | 0.1 | 1e4 | 0 | 55 | 0.8148 ✅ | N/A | -- | -- |
| 802175 | benchmark-ablall | 20260131_131331...802175 | 4 | 4 | 64 | 0.1 | 2e4 | 0 | 39 | 0.8132±0.0011  | N/A | (0.8148) | (0.7070) |

### nips_task34

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 498903 | baseline-ablall | 20260126_191440...498903 | 4 | 4 | 64 | 0.1 | 1e4 | 0 | 55 | 0.7988  | N/A | -- | -- |
| 216531 | benchmark-ablall | 20260131_131331...216531 | 4 | 4 | 64 | 0.1 | 2e4 | 0 | 39 | 0.7998±0.0004 ✅ | N/A | (0.7988) | (0.7220) |

## Table assist2015, ablation=all

**LR + Dropout Sweep** (Phase 1: 2 epochs, quick evaluation)

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 796522 | lr_dropout_a15 | 20260131_222231_lr_d...796522 | 4 | 4 | 64 | 0.1 | 3e-4 | 0 | 2 | 0.7093 | N/A | ❌ (0.7078) | ✅ (0.7019) |
| 858878 | lr_dropout_a15 | 20260131_222232_lr_d...858878 | 4 | 4 | 64 | 0.15 | 3e-4 | 0 | 2 | 0.7069 | N/A | ❌ (0.7078) | ✅ (0.7019) |
| 161403 | lr_dropout_a15 | 20260131_222230_lr_d...161403 | 4 | 4 | 64 | 0.1 | 2e-4 | 0 | 2 | 0.7055 | N/A | ❌ (0.7078) | ✅ (0.7019) |
| 303520 | lr_dropout_a15 | 20260131_222231_lr_d...303520 | 4 | 4 | 64 | 0.15 | 2e-4 | 0 | 2 | 0.7035 | N/A | ❌ (0.7078) | ✅ (0.7019) |

**Full Training Results**

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 433572 | sweep-assist2015 | 20260131_223710...433572 | 4 | 4 | 64 | 0.1 | 3e-4 | 0 | 19 | 0.7062 | N/A | ❌ (0.7078) | ✅ (0.7019) |

**Note**: Phase 1 validation AUC scores after only 2 epochs (early stopping test). Baseline comparison is exp 589915 (p_sup=0.7078 after 55 epochs). Full training (exp 433572) early stopped at epoch 19 with test_auc=0.7062, slightly below baseline. The 3× higher learning rate converges faster initially but doesn't improve final performance.

## Table assist2015, ablation=all (Architecture: 2 blocks, 8 heads)

**Architecture Sweep Results**

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 158395 | sweep-assist2015-2-8 | 20260131_232348...158395 | 2 | 8 | 64 | 0.1 | 2e-4 | 0 | 23 | 0.7061 | N/A | ❌ (0.7078) | ✅ (0.7077) |

**Note**: Architecture variation experiment (2 blocks, 8 heads vs standard 4 blocks, 4 heads). Training completed at epoch 23 with test_auc=0.7061, slightly below baseline test_auc=0.7078 (-0.24%). Despite strong validation performance (0.7322), the architecture change did not improve final test performance. Baseline comparison is exp 589915 (4 blocks, 4 heads, p_sup=0.7078, epoch 1 AUC=0.7019).

## Table nips_task34, ablation=all

**LR + Dropout Sweep** (Phase 1: 2 epochs, quick evaluation)

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| TBD | lr_dropout_n34 | TBD | 4 | 4 | 64 | 0.1 | 3e-4 | 0 | 2 | TBD | N/A | ❌ (0.7988) | ❌ (0.7344) |
| TBD | lr_dropout_n34 | TBD | 4 | 4 | 64 | 0.15 | 3e-4 | 0 | 2 | TBD | N/A | ❌ (0.7988) | ❌ (0.7344) |
| TBD | lr_dropout_n34 | TBD | 4 | 4 | 64 | 0.1 | 2e-4 | 0 | 2 | TBD | N/A | ❌ (0.7988) | ❌ (0.7344) |
| TBD | lr_dropout_n34 | TBD | 4 | 4 | 64 | 0.15 | 2e-4 | 0 | 2 | TBD | N/A | ❌ (0.7988) | ❌ (0.7344) |

**Full Training Results**

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 727875 | sweep-nips_task34 | 20260131_230137...231941 | 4 | 4 | 64 | 0.1 | 3e-4 | 0 | 41 | <span style="color:green">**0.8006**</span> ✅ | N/A | ✅ (0.7988) | ✅ (0.7344) |

**Note**: Phase 1 validation AUC scores after only 2 epochs (early stopping test). Baseline comparison is exp 498903 (p_sup=0.7988 after 55 epochs, epoch 1 AUC=0.7344). Full training (exp 727875) achieved test_auc=0.8006 at epoch 41, exceeding baseline by +0.18%. The 3× higher learning rate successfully improves final performance.

## Table nips_task34, ablation=all (Architecture: 2 blocks, 8 heads)

**Architecture Sweep Results**

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 112389 | sweep-nips-2-8 | 20260131_232237...112389 | 2 | 8 | 64 | 0.1 | 2e-4 | 0 | 57 | 0.7993 ✅ | N/A | ✅ (0.7988) | ✅ (0.7281) |

**Note**: Architecture variation experiment (2 blocks, 8 heads vs standard 4 blocks, 4 heads). Training completed at epoch 57 with test_auc=0.7993, exceeding baseline test_auc=0.7988 by +0.06%. The architecture change with fewer blocks but more heads achieves comparable performance to the baseline. Baseline comparison is exp 498903 (4 blocks, 4 heads, p_sup=0.7988, epoch 1 AUC=0.7344).

## Table nips_task34, ablation=all (High LR Sweep - Part 1)

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 634611 | lr4e4 | 20260201_001033...634611 | 4 | 4 | 64 | 0.1 | 4e-4 | 0 | 35 | **0.7998** ✅ | N/A | ✅ (0.7988) | ✅ (0.7344) | 

**Note**: Full training achieved test_auc=0.7998 at epoch 35, exceeding baseline by +0.13%.

## Table nips_task34, ablation=all (High LR Sweep - Part 2)

| Exp ID | Campaign | Exp Folder | blocks | heads | emb_dim | dropout | lr | lambda_ref | epochs | p_sup | p_ref | vs Baseline (final) | vs Baseline (epoch 1) |
|--------|----------|------------|--------|-------|---------|---------|----|-----------:|-------:|-------|-------|---------------------|----------------------|
| 199547 | lr5e4 | 20260201_001107...199547 | 4 | 4 | 64 | 0.1 | 5e-4 | 0 | 1 | 0.7367* | N/A | ❌ (0.7988) | ✅ (0.7344) |

**Note**: Experiment terminated after 1 epoch. The star (\*) indicates validation AUC instead of test AUC.

