#!/usr/bin/env python3
"""
Centralized launcher for the ECTEL 2026 paper visualisation pipeline.

Generates the Situational Instruction plots for every requested gtransformer
experiment.  The three scripts it invokes are:

  1. examples/results/generate_roster_plots_gtransformer.py
       Cognitive Roster: 2-D quadrant scatter, 3-D trajectory scatter,
       Student x Skill heatmap, and animated GIFs.

  2. examples/results/generate_attractor_plots_gtransformer.py
       Attractor covariance ellipses (one per learning-situation quadrant).

  3. examples/results/generate_attractor_dynamics_plots.py
       KDE density contours, return/lag maps, state-transition graphs.

All plots are written to a dedicated plots_ectel/ directory placed at the
dataset level beside the existing plots/ directory used by the MDPI pipeline:

    experiments/<campaign>/gtransformer/<dataset>/plots_ectel/

The script uses fold 0 as the representative run (configurable via --fold).
It will fail loudly if required trajectory files are absent rather than
silently skipping, because missing files indicate an incomplete experiment.

Usage examples (run inside the Docker container with the virtual environment):

    # All datasets in the most recent benchpaper campaign
    python examples/run_ectel_paper.py

    # Specific campaign
    python examples/run_ectel_paper.py --campaign 20260202_222258_benchpaper_assist2009_mdpipaper_893468

    # Specific dataset only
    python examples/run_ectel_paper.py --campaign 20260202_222258_benchpaper_assist2009_mdpipaper_893468 --dataset assist2009

    # Override fold used as representative
    python examples/run_ectel_paper.py --campaign 20260202_222258_benchpaper_assist2009_mdpipaper_893468 --fold 1

    # Skip animated GIFs (faster)
    python examples/run_ectel_paper.py --no-gif
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# ── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Datasets that the ECTEL pipeline supports (subset that have traj_ files)
ECTEL_DATASETS = [
    "assist2009",
    "assist2015",
    "bridge2algebra2006",
    "nips_task34",
    "algebra2005",
]

REQUIRED_FILES = ["traj_rate.csv", "traj_initmastery.csv"]

ECTEL_SCRIPTS = [
    {
        "name": "Cognitive Roster (2-D scatter, 3-D scatter, heatmap, GIFs)",
        "script": "examples/results/generate_roster_plots_gtransformer.py",
        "gif_flag": True,   # pass --gif when gif=True
    },
    {
        "name": "Attractor covariance ellipses",
        "script": "examples/results/generate_attractor_plots_gtransformer.py",
        "gif_flag": False,
    },
    {
        "name": "Attractor dynamics (KDE, return-map, transition graph)",
        "script": "examples/results/generate_attractor_dynamics_plots.py",
        "gif_flag": False,
    },
]


# ── experiment-folder discovery (mirrors run_benchmarks_paper.py logic) ──────

def _find_campaigns(campaign_arg: Optional[str]) -> list:  # list[Path]
    """Return a list of campaign directories to search, newest first."""
    base = PROJECT_ROOT / "experiments"

    if campaign_arg:
        pattern = campaign_arg if "*" in campaign_arg else f"*{campaign_arg}*"
        found = sorted(base.glob(pattern))
        if not found:
            raise FileNotFoundError(
                f"No campaign directory matches pattern '{pattern}' under {base}"
            )
        return list(reversed(found))

    # No campaign specified: use the most recent timestamped directory.
    timestamped = [
        d for d in base.iterdir()
        if d.is_dir()
        and len(d.name) >= 15
        and d.name[:8].isdigit()
        and d.name[8] == "_"
        and d.name[9:15].isdigit()
    ]
    if not timestamped:
        raise FileNotFoundError(f"No timestamped campaign directory found under {base}")
    return [sorted(timestamped, key=lambda p: p.name)[-1]]


def find_fold_dir(campaign: Path, dataset: str, fold: int) -> Optional[Path]:
    """
    Return the fold directory for gtransformer/<dataset>/fold_<fold>_* inside
    a given campaign, validated against config.json.
    Returns None if nothing matching is found.
    """
    target = campaign / "gtransformer" / dataset
    if not target.exists():
        return None

    candidates = list(target.glob(f"fold_{fold}_*"))
    if not candidates:
        return None

    valid = []
    for c in candidates:
        cfg_path = c / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        inp = cfg.get("input", {})
        trn = cfg.get("train_config", {})
        dfl = cfg.get("defaults", {})
        c_data  = inp.get("dataset",  trn.get("dataset",  dfl.get("dataset")))
        c_fold  = inp.get("fold",     trn.get("fold",     dfl.get("fold")))
        c_model = inp.get("model",    trn.get("model",    dfl.get("model")))
        if (str(c_data) == dataset
                and str(c_fold) == str(fold)
                and str(c_model) == "gtransformer"):
            valid.append((c.stat().st_mtime, c))

    if not valid:
        return None
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1]


# ── script runner ─────────────────────────────────────────────────────────────

def run_script(script_rel: str, run_dir: Path, output_dir: Path,
                gif: bool, timeout: int) -> Tuple[bool, str]:
    """
    Execute a single plot-generation script.
    Returns (success, message).
    """
    script_path = PROJECT_ROOT / script_rel
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    cmd = [
        sys.executable, str(script_path),
        "--run_dir",    str(run_dir),
        "--output_dir", str(output_dir),
    ]
    info = ECTEL_SCRIPTS[[s["script"] for s in ECTEL_SCRIPTS].index(script_rel)]
    if gif and info.get("gif_flag"):
        cmd.append("--gif")

    try:
        result = subprocess.run(
            cmd, check=False, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            snippet = (result.stderr or result.stdout or "")[:300].strip()
            return False, f"exit code {result.returncode}: {snippet}"
        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ECTEL 2026 Situational Instruction plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help=(
            "Campaign directory name or glob pattern under experiments/. "
            "If omitted the most recent timestamped campaign is used."
        ),
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        metavar="DATASET",
        help=(
            f"One or more dataset names to process. "
            f"Defaults to all: {', '.join(ECTEL_DATASETS)}."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to use as the representative run (default: 0).",
    )
    parser.add_argument(
        "--fallback_folds",
        action="store_true",
        default=True,
        help=(
            "If fold --fold is not found, try folds 1-4 before giving up "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--no-gif",
        dest="gif",
        action="store_false",
        default=True,
        help="Skip animated GIF generation (faster).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-script timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Override output directory. If not set, plots are written to "
            "plots_ectel/ inside the dataset directory of the campaign."
        ),
    )
    args = parser.parse_args()

    datasets = args.dataset or ECTEL_DATASETS

    # Validate dataset names
    unknown = [d for d in datasets if d not in ECTEL_DATASETS]
    if unknown:
        parser.error(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Valid options: {', '.join(ECTEL_DATASETS)}"
        )

    # Locate campaigns
    try:
        campaigns = _find_campaigns(args.campaign)
    except FileNotFoundError as exc:
        sys.exit(f"[ERROR] {exc}")

    print("=" * 70)
    print("ECTEL 2026 — Situational Instruction Visualisation Pipeline")
    print("=" * 70)
    print(f"  Campaign search pattern : {args.campaign or '(most recent)' }")
    print(f"  Campaigns found         : {len(campaigns)}")
    print(f"  Datasets                : {', '.join(datasets)}")
    print(f"  Representative fold     : {args.fold}")
    print(f"  Animated GIFs           : {'yes' if args.gif else 'no'}")
    print(f"  Per-script timeout      : {args.timeout}s")
    print("=" * 70)

    total_generated = total_skipped = total_failed = 0

    for campaign in campaigns:
        print(f"\nCampaign: {campaign.name}")

        for dataset in datasets:
            print(f"\n  Dataset: {dataset}")

            # Find the representative fold directory
            fold_dir = find_fold_dir(campaign, dataset, args.fold)
            if fold_dir is None and args.fallback_folds:
                for fb in range(1, 5):
                    fold_dir = find_fold_dir(campaign, dataset, fb)
                    if fold_dir is not None:
                        print(
                            f"    [INFO] Fold {args.fold} not found; "
                            f"using fold {fb} as representative."
                        )
                        break

            if fold_dir is None:
                print(
                    f"    [SKIP] No valid gtransformer/{dataset}/fold_* directory "
                    f"found in campaign '{campaign.name}'."
                )
                total_skipped += len(ECTEL_SCRIPTS)
                continue

            print(f"    Representative fold : {fold_dir.name}")

            # Check that required trajectory files are present
            missing = [f for f in REQUIRED_FILES if not (fold_dir / f).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Required trajectory file(s) missing in {fold_dir}: "
                    f"{', '.join(missing)}. "
                    "Re-run evaluation with dual_eval enabled to generate them."
                )

            # Resolve output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                # dataset_dir is campaign/gtransformer/<dataset>/
                dataset_dir = fold_dir.parent
                output_dir = dataset_dir / "plots_ectel"

            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"    Output              : {output_dir}")

            # Run each script
            for script_info in ECTEL_SCRIPTS:
                name = script_info["name"]
                print(f"\n    [RUN] {name} ...")
                ok, msg = run_script(
                    script_info["script"],
                    fold_dir,
                    output_dir,
                    gif=args.gif,
                    timeout=args.timeout,
                )
                if ok:
                    print(f"      [OK]")
                    total_generated += 1
                else:
                    print(f"      [FAIL] {msg}")
                    total_failed += 1

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Generated : {total_generated}")
    print(f"  Skipped   : {total_skipped}")
    print(f"  Failed    : {total_failed}")
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
