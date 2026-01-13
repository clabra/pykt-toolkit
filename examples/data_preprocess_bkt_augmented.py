import os
import sys
import argparse
from pykt.preprocess.split_datasets_bkt import main as split_bkt
from pykt.preprocess.assist2009_preprocess_bkt import read_data_from_csv as assist2009_bkt_reader

dname2paths = {
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed_bkt_augmented.csv"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="assist2009")
    parser.add_argument("-m", "--min_seq_len", type=int, default=3)
    parser.add_argument("-l", "--maxlen", type=int, default=200)
    parser.add_argument("-k", "--kfold", type=int, default=5)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    configf = os.path.join(project_root, "configs/data_config.json")
    
    if args.dataset_name not in dname2paths:
        print(f"Error: Dataset {args.dataset_name} not supported by BKT augmented preprocessor.")
        sys.exit(1)

    readf = dname2paths[args.dataset_name]
    # Ensure raw path is absolute
    if readf.startswith("../"):
        readf = os.path.join(project_root, readf[3:])
    
    # Target directory for BKT augmented data
    dname = os.path.join(os.path.dirname(readf), "bkt_augmented")
    os.makedirs(dname, exist_ok=True)
    
    writef = os.path.join(dname, "data_bkt.txt")
    
    print(f"Starting BKT augmented preprocessing for {args.dataset_name}")
    print(f"Source: {readf}")
    print(f"Output: {dname}")

    # 1. Convert CSV to 8-line data.txt
    if args.dataset_name == "assist2009":
        assist2009_bkt_reader(readf, writef)
    else:
        print("Only assist2009 is supported currently.")
        sys.exit(1)

    # 2. Split and Generate Sequences
    split_bkt(dname, writef, args.dataset_name, configf, args.min_seq_len, args.maxlen, args.kfold)
    
    print("\n" + "="*50)
    print(f"SUCCESS: BKT Augmented dataset created as '{args.dataset_name}_bkt'")
    print("="*50)

if __name__ == "__main__":
    main()
