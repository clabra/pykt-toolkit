# Scripts used for ECTEL paper (Situational Instruction plots, etc.)

All plots in this section are written to a dedicated `plots_ectel/` directory at the dataset level, separate from the `plots/` directory used by the MDPI pipeline:

```
experiments/<campaign>/gtransformer/<dataset>/plots_ectel/
```

Every script reads `traj_rate.csv` and `traj_initmastery.csv` from the representative fold directory (generated during evaluation). Students are partitioned into four learning-situation quadrants defined by the medians of initial mastery ($P(L_0)$) and learning rate ($P(T)$): *foundational* (low $P(L_0)$, low $P(T)$), *consolidating* (high $P(L_0)$, low $P(T)$), *emerging* (low $P(L_0)$, high $P(T)$), and *advancing* (high $P(L_0)$, high $P(T)$).

---

## Centralized launcher

`examples/run_ectel_paper.py` runs all seven individual scripts in sequence for one or more datasets. It is independent of `run_benchmarks_paper.py`.

```bash
# All datasets in the most recent benchpaper campaign
python examples/run_ectel_paper.py

# Specific campaign and dataset
python examples/run_ectel_paper.py \
    --campaign 20260202_222258_benchpaper_assist2009_mdpipaper_893468 \
    --dataset assist2009

# Skip animated GIFs to reduce runtime
python examples/run_ectel_paper.py \
    --campaign 20260202_222258_benchpaper_assist2009_mdpipaper_893468 \
    --dataset assist2009 --no-gif

# Use fold 1 as the representative run instead of fold 0
python examples/run_ectel_paper.py \
    --campaign 20260202_222258_benchpaper_assist2009_mdpipaper_893468 \
    --dataset assist2009 --fold 1
```

### Launcher parameters

| Parameter | Default | Description |
|---|---|---|
| `--campaign` | latest | Campaign directory name or glob pattern. If omitted, the most recently created campaign under `experiments/` is used. |
| `--dataset` | all found | One or more dataset names (space-separated). |
| `--fold` | `0` | Index of the representative fold to use. |
| `--fallback_folds` | `1 2 3 4` | Fallback fold indices tried in order when the primary fold directory is not found. |
| `--no-gif` | off | Skip animated GIF generation (passed through to Script 1). |
| `--timeout` | `300` | Per-script timeout in seconds. |
| `--output_dir` | auto | Override the output directory (otherwise `plots_ectel/` inside the dataset directory). |

---

## Script 1 — Cognitive Roster

**Script**: `examples/results/generate_roster_plots_gtransformer.py`

Selects one representative student per quadrant and visualises their learning trajectory in the $(P(L_0), P(T))$ space. The representative maximises a composite score that balances proximity to the quadrant centroid with number of recorded interactions (weights controlled by `--weight_centroid`). Candidate trajectory points are evaluated every `--timestep` interactions and plotted only when the Euclidean distance from the last plotted point exceeds an auto-computed `min_move` threshold; skipped candidates are absorbed into the preceding point, whose marker size grows to encode dwell time. Points with $P(T) >$ `--rate_max` are treated as outliers and excluded before plotting; the script emits a warning when their fraction exceeds `--rate_outlier_warn_pct`.

### Output files

| File | Description |
|---|---|
| `roster_2d_{quad}_893468.png` | 2-D trajectory scatter in the $P(L_0) \times P(T)$ plane with quadrant background regions. One file per quadrant (4 total). |
| `roster_traj_{quad}_893468.gif` | Animated GIF of the same trajectory unfolding over time. One per quadrant (4 total, only with `--gif`). |
| `roster_3d_situations_893468.png` | Combined 3-D scatter of all four representatives in the $(P(L_0), P(T), t)$ space. |
| `roster_heatmap_students_893468.png` | Student × Skill heatmap of mean mastery per representative. |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | required | Path to the fold directory containing `traj_rate.csv` and `traj_initmastery.csv`. |
| `--output_dir` | `--run_dir` | Directory where plots are saved. |
| `--timestep` | `5` | Subsample stride: evaluate a candidate point every N interactions. |
| `--min_interactions` | `20` | Minimum number of interactions for a student to be eligible as a quadrant representative. |
| `--weight_centroid` | `0.5` | Weight given to centroid proximity when scoring representative candidates; the complement (`1 - weight`) goes to interaction count. Both are min-max normalised before weighting. |
| `--count_iqr_threshold` | `1.5` | IQR multiplier for interaction-count outlier detection; students above $Q_3 + k \cdot \text{IQR}$ are excluded from representative selection. |
| `--min_move` | auto | Override the auto-computed minimum Euclidean distance (in the $[0,1]^2$ space) required to plot a candidate point. If unset, derived by binary search so the busiest representative has at most `--max_points` plotted points. |
| `--max_points` | `20` | Target maximum number of plotted points per trajectory; used only when `--min_move` is not set. |
| `--rate_max` | `0.5` | Upper bound on the $P(T)$ axis. Points with $P(T) >$ `rate_max` are excluded from the 2-D plots and the axis is capped to this value. |
| `--rate_outlier_warn_pct` | `5.0` | Percentage threshold for the rate-axis outlier check. A `UserWarning` is emitted when more than this fraction of interactions (globally, or per representative student) exceed `--rate_max`, suggesting the clipped points may not be genuine outliers. |
| `--gif` | off | Also save animated GIFs. |
| `--gif_fps` | `2` | Frames per second for the GIF. |
| `--gif_hold` | `4` | Number of extra repeated frames at the end of the GIF so the final state is visible. |

### Individual invocation

```bash
python examples/results/generate_roster_plots_gtransformer.py \
    --run_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/fold_0_377291 \
    --output_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/plots_ectel \
    --rate_max 0.5 --rate_outlier_warn_pct 5.0 --gif
```

---

## Script 2 — Attractor covariance ellipses

**Script**: `examples/results/generate_attractor_plots_gtransformer.py`

For each quadrant representative, draws the student's full trajectory as a faded scatter overlaid with 1-sigma and 2-sigma covariance ellipses centred on the mean position, visualising the shape and spread of the attractor orbit in the $(P(L_0), P(T))$ plane.

### Output files

| File | Description |
|---|---|
| `attractor_ellipse_{quad}_893468.png` | Covariance ellipse plot for the representative of each quadrant. One file per quadrant (4 total). |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | required | Path to the fold directory. |
| `--output_dir` | `--run_dir` | Directory where plots are saved. |
| `--min_interactions` | `20` | Minimum interactions for representative eligibility. |
| `--weight_centroid` | `0.3` | Centroid-proximity weight in representative scoring (note: different default from Script 1). |
| `--count_iqr_threshold` | `1.5` | IQR multiplier for interaction-count outlier exclusion. |
| `--uid_suffix` | `893468` | Suffix appended to all output filenames (matches the campaign seed). |

### Individual invocation

```bash
python examples/results/generate_attractor_plots_gtransformer.py \
    --run_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/fold_0_377291 \
    --output_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/plots_ectel
```

---

## Script 3 — Attractor dynamics

**Script**: `examples/results/generate_attractor_dynamics_plots.py`

Produces three complementary families of attractor visualisations per quadrant, pooling all students assigned to that quadrant.

### Output files

| File | Description |
|---|---|
| `attractor_kde_{quad}_893468.png` | KDE density contour map of the joint distribution of $(P(L_0), P(T))$ with the representative trajectory overlaid. |
| `attractor_returnmap_{quad}_893468.png` | Lag-1 return map: $P(T)_{t+1}$ vs $P(T)_t$ and $P(L_0)_{t+1}$ vs $P(L_0)_t$ in two sub-panels. Convergence to a fixed point appears as a cluster near the diagonal. |
| `attractor_transgraph_{quad}_893468.png` | State-transition graph: the $(P(L_0), P(T))$ space is divided into a coarse grid; directed edges show the most frequent pairwise transitions within the quadrant. |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | required | Path to the fold directory. |
| `--output_dir` | `--run_dir` | Directory where plots are saved. |
| `--min_interactions` | `20` | Minimum interactions for representative eligibility. |
| `--weight_centroid` | `0.3` | Centroid-proximity weight in representative scoring. |
| `--count_iqr_threshold` | `1.5` | IQR multiplier for interaction-count outlier exclusion. |
| `--grid_n` | `6` | Number of cells per axis for the state-transition grid ($n \times n$ total cells). |
| `--top_k` | `15` | Maximum number of directed edges shown in the transition graph (highest-frequency edges are kept). |
| `--uid_suffix` | `893468` | Suffix appended to all output filenames. |

### Individual invocation

```bash
python examples/results/generate_attractor_dynamics_plots.py \
    --run_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/fold_0_377291 \
    --output_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/plots_ectel
```


---

## Script 4 — Learning situation distribution over time

**Script**: `examples/results/generate_situation_distribution_plots.py`

At each interaction snapshot (every `--stride` interactions), assigns each student to a learning situation based on the mean of their parameters accumulated up to that snapshot, and plots how the distribution evolves across the full session. Population medians for $P(L_0)$ and $P(T)$ are computed once on the full dataset and used as fixed quadrant boundaries at every snapshot.

### Output files

| File | Description |
|---|---|
| `situation_dist_stacked_<uid_suffix>.png` | Stacked area chart: fraction of students in each learning situation vs. interaction index. |
| `situation_dist_counts_<uid_suffix>.png` | Stacked bar chart: absolute student count per learning situation vs. interaction index. |
| `situation_dist_heatmap_<uid_suffix>.png` | Student × snapshot heatmap coloured by assigned learning situation; students sorted by dominant quadrant. |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | required | Fold directory containing `traj_rate.csv` and `traj_initmastery.csv`. |
| `--output_dir` | `--run_dir` | Directory where plots are saved. |
| `--stride` | `10` | Snapshot interval in number of interactions. |
| `--min_interactions` | `5` | Students with fewer total interactions are excluded. |
| `--max_interactions` | p90 | Cap trajectories at this length. Defaults to the 90th percentile of student interaction counts. |
| `--uid_suffix` | `893468` | Suffix appended to all output filenames. |

### Individual invocation

```bash
python examples/results/generate_situation_distribution_plots.py \
    --run_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/fold_0_377291 \
    --output_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/plots_ectel \
    --stride 10
```

---

## Script 5 — Learning situation dynamics (counts + churn)

**Script**: `examples/results/generate_situation_dynamics_plot.py`

Two-panel figure: the top panel shows the stacked bar distribution of students across learning situations at each interaction snapshot, with per-situation trend lines overlaid. The bottom panel shows *transition churn* — the fraction of students whose assigned learning situation changed relative to the previous snapshot — providing direct evidence that situation assignments are not static.

### Output files

| File | Description |
|---|---|
| `situation_dynamics_<uid_suffix>.png` | Two-panel figure: stacked bar counts with trend lines (top) and churn percentage curve (bottom). |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | required | Fold directory containing `traj_rate.csv` and `traj_initmastery.csv`. |
| `--output_dir` | `--run_dir` | Directory where plots are saved. |
| `--stride` | `10` | Snapshot interval in number of interactions. |
| `--min_interactions` | `5` | Students with fewer total interactions are excluded. |
| `--max_interactions` | p90 | Cap trajectory length. Defaults to the 90th percentile of student interaction counts. |
| `--smooth_window` | `5` | Uniform smoothing window applied to trend lines and churn curve. Set to `1` to disable. |
| `--uid_suffix` | `893468` | Suffix appended to all output filenames. |

### Individual invocation

```bash
python examples/results/generate_situation_dynamics_plot.py \
    --run_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/fold_0_377291 \
    --output_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/plots_ectel
```

---

## Script 6 — Learning situation transitions (distribution + transition matrix)

**Script**: `examples/results/generate_situation_transitions_plot.py`

Two-panel figure combining the stacked bar distribution (left) with a 4×4 gross transition matrix (right). The matrix counts every individual A→B reassignment across all consecutive snapshot pairs — unlike churn, opposing flows do not cancel. A text box in the left panel reports the total fraction of student-snapshot pairs that involved a situation change.

### Output files

| File | Description |
|---|---|
| `situation_transitions_<uid_suffix>.png` | Stacked bar counts with trend lines (left) and gross transition probability matrix (right). |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | required | Fold directory containing `traj_rate.csv` and `traj_initmastery.csv`. |
| `--output_dir` | `--run_dir` | Directory where plots are saved. |
| `--stride` | `10` | Snapshot interval in number of interactions. |
| `--min_interactions` | `5` | Students with fewer total interactions are excluded. |
| `--max_interactions` | p90 | Cap trajectory length. Defaults to the 90th percentile of student interaction counts. |
| `--smooth_window` | `3` | Smoothing window for bar trend lines. |
| `--uid_suffix` | `893468` | Suffix appended to all output filenames. |

### Individual invocation

```bash
python examples/results/generate_situation_transitions_plot.py \
    --run_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/fold_0_377291 \
    --output_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/plots_ectel
```

---

## Script 7 — Learning situation entry vs. exit

**Script**: `examples/results/generate_situation_entry_exit_plot.py`

Compares each student's learning situation at the start of their session (entry: mean over the first `--window` interactions) against their situation at the end (exit: mean over the last `--window` interactions). Only students with at least `2 × window` interactions are included so the two windows do not overlap.

Produces two separate output files:

- An **alluvial (Sankey) diagram** where ribbons connect entry to exit columns; ribbon width encodes the number of students, making cross-situation flows directly visible even when opposing flows cancel in aggregate.
- A **transition probability heatmap** showing the fraction of students from each entry situation who ended in each exit situation (each row sums to 100%).

### Output files

| File | Description |
|---|---|
| `situation_alluvial_<uid_suffix>.png` | Alluvial diagram: ribbon width proportional to student count; ribbons coloured by entry situation. |
| `situation_transition_matrix_<uid_suffix>.png` | 4×4 heatmap of entry→exit transition percentages; raw counts in parentheses; each row sums to 100%. |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--run_dir` | required | Fold directory containing `traj_rate.csv` and `traj_initmastery.csv`. |
| `--output_dir` | `--run_dir` | Directory where plots are saved. |
| `--window` | `30` | Number of interactions used for the entry and exit windows. Students with fewer than `2 × window` interactions are excluded. |
| `--uid_suffix` | `893468` | Suffix appended to all output filenames. |

### Individual invocation

```bash
python examples/results/generate_situation_entry_exit_plot.py \
    --run_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/fold_0_377291 \
    --output_dir experiments/20260202_222258_benchpaper_assist2009_mdpipaper_893468/gtransformer/assist2009/plots_ectel
```
