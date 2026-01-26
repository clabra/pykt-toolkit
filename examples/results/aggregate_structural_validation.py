
import os
import json
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign_dir", type=str, required=True, help="Path to campaign directory containing validation/ folder")
    args = parser.parse_args()

    validation_dir = Path(args.campaign_dir) / "validation"
    if not validation_dir.exists():
        print(f"Error: {validation_dir} does not exist.")
        return

    json_files = list(validation_dir.glob("structural_encoding_fold_*.json"))
    if not json_files:
        print(f"No results found in {validation_dir}")
        return

    all_results = []
    for jf in json_files:
        with open(jf, 'r') as f:
            all_results.append(json.load(f))

    print(f"\n=== Aggregating H1.1 Results across {len(all_results)} folds ===")
    
    report = {
        "dataset": all_results[0]["dataset"],
        "num_folds": len(all_results),
        "results": {}
    }

    for construct in ["l0", "t"]:
        report["results"][construct] = {
            "fidelity_r2": [],
            "fidelity_pearson": [],
            "selectivity_std": [],
            "selectivity_strict": []
        }
        
        for res in all_results:
            data = res["h1_structural_encoding"][construct]
            report["results"][construct]["fidelity_r2"].append(data["fidelity"]["r2"])
            report["results"][construct]["fidelity_pearson"].append(data["fidelity"]["pearson_r"])
            report["results"][construct]["selectivity_std"].append(data["selectivity"]["standard_delta_r2"])
            report["results"][construct]["selectivity_strict"].append(data["selectivity"]["strict_delta_r2"])

        print(f"\n--- {construct.upper()} ---")
        for metric, values in report["results"][construct].items():
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric:<20}: {mean:.4f} \u00b1 {std:.4f}")

    output_path = validation_dir / "structural_encoding_aggregated.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nAggregated results saved to: {output_path}")

if __name__ == "__main__":
    main()
