
"""
Run Prior-guided Bayesian Optimization (PiBO) for Synthetic Priors.
Features a unified toggle to run across Buchwald, Suzuki, or Direct Arylation.
Pulls a single prior set and runs it across multiple Beta configurations,
saving the results into Beta-specific output folders.
"""

import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# PATH INJECTION FOR PiBO
# ---------------------------------------------------------
# 1. Get the directory of the current script (...\2_Synthetic_Prior_BO)
_script_dir = Path(__file__).resolve().parent

# 2. Get the parent directory (...\External_LLMpiBO_code_base)
_base_dir = _script_dir.parent

# 3. Add base_dir to sys.path so Python can see the PiBO folder inside it
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

# 4. Import directly from the PiBO package
try:
    from PiBO import PiBOCampaignRunner
except ModuleNotFoundError:
    # Fallback just in case it's nested or named slightly differently
    from PiBO.PiBO import PiBOCampaignRunner

import warnings
from botorch.exceptions.warnings import NumericsWarning
warnings.filterwarnings("ignore", category=NumericsWarning)

# ---------------------------------------------------------
# DATASET CONFIGURATIONS
# ---------------------------------------------------------
DATASET_CONFIGS = {
    "Buchwald": {
        "param_columns": ["additive", "aryl_halide", "base", "ligand"],
    },
    "Suzuki": {
        "param_columns": ["Electrophile_SMILES", "Nucleophile_SMILES", "Ligand_SMILES", "Base_SMILES", "Solvent_SMILES"],
    },
    "Direct": {
        "param_columns": ["Base_SMILES", "Ligand_SMILES", "Solvent_SMILES"],
    }
}

def load_prior_set(path: Path, num_params: int) -> list:
    """Reads CSV of priors and returns tuples of categorical indices."""
    priors = []
    with open(path) as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return []
        
        # Detect header
        try:
            float(first[0])
            priors.append(tuple(int(float(x)) for x in first[:num_params]))
        except ValueError:
            pass 
            
        for row in reader:
            if len(row) >= num_params:
                priors.append(tuple(int(float(x)) for x in row[:num_params]))
    return priors

def main():
    # ---------------------------------------------------------
    # CONFIGURATION VARIABLES
    # ---------------------------------------------------------
    # Easily swap datasets here: "Buchwald", "Suzuki", or "Direct"
    DATASET_NAME = "Suzuki"  
    
    # Updated to run the Synthetic Prior variants
    PRIOR_VARIANTS = ["ideal", "misleading"]
    
    N_PRIORS = [1,3,5,10]         # Run all batch sizes
    MAX_SETS = 30                 # Number of prior sets (set_0 to set_29)
    BETAS = [100]                 # PiBO decay weights (1=soft, 100=strict)
    PERCENTILE = 2.0              # Target top 2% of yield
    MAX_ITERATIONS = 11           # Safety limit for iterations
    # ---------------------------------------------------------

    if DATASET_NAME not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{DATASET_NAME}'. Choose from: {list(DATASET_CONFIGS.keys())}")
        
    config = DATASET_CONFIGS[DATASET_NAME]
    param_columns = config["param_columns"]

    # Input data path dynamically routed to centralized Data folder
    data_dir = _base_dir / "Data" / DATASET_NAME
    data_path = data_dir / "processed.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

    df = pd.read_csv(data_path)
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce").fillna(0.0)

    # Standardize parameter values to strings for BayBE using the dynamic columns
    param_values = []
    for c in param_columns:
        df[c] = df[c].apply(lambda x: str(int(float(x))) if pd.notna(x) else x)
        valid_strs = sorted(list(df[c].dropna().unique()), key=int)
        param_values.append(valid_strs)
    
    # Calculate global target yield threshold
    target_yield = float(np.percentile(df["yield"].values, 100 - PERCENTILE))
    print(f"Target yield (top {PERCENTILE}%) for {DATASET_NAME}: {target_yield:.2f}\n")

    # Initialize Runner
    runner = PiBOCampaignRunner(
        param_names=param_columns,
        param_values=param_values,
        df=df,
        yield_col="yield",
    )

    # Loop through prior variants
    for prior_model in PRIOR_VARIANTS:
        print(f"\n=======================================================")
        print(f" STARTING PiBO CAMPAIGN: {DATASET_NAME} | {prior_model.upper()}")
        print(f"=======================================================\n")
        
        for set_num in range(MAX_SETS):
            print(f"=== Evaluating Set {set_num} ({prior_model.upper()}) ===")
            
            for current_n in N_PRIORS:
                # 1. Load the prior ONCE for this N and set number
                prior_path = _script_dir / "priors" / DATASET_NAME / prior_model / f"n_priors_{current_n}" / f"set_{set_num}.csv"
                
                if not prior_path.exists():
                    print(f"    [Skip] Could not find {prior_path.relative_to(_script_dir)}")
                    continue
                    
                raw_priors = load_prior_set(prior_path, len(param_columns))
                
                prior_set = []
                for p in raw_priors:
                    if all(str(int(p[i])) in param_values[i] for i in range(len(p))):
                        prior_set.append(p)
                
                prior_set_fast = set(prior_set)
                
                # Base output directory within piBO_runs
                out_base = _script_dir / "piBO_runs" / DATASET_NAME / prior_model / f"n_priors_{current_n}" / f"set_{set_num}"
                out_base.mkdir(parents=True, exist_ok=True)

                # 2. Run the optimization for each Beta using the same loaded prior
                for beta in BETAS:
                    results_dir = out_base / f"beta_{beta}"
                    results_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Execute BO Campaign
                    iters_df, summary = runner.run(
                        prior_set=prior_set,
                        beta=beta,
                        target_yield=target_yield,
                        max_iterations=MAX_ITERATIONS,
                        results_dir=results_dir,
                        set_num=set_num,
                        n_priors=current_n,
                    )
                    
                    in_prior_list = []
                    for _, row in iters_df.iterrows():
                        pt = tuple(int(float(row[col])) for col in param_columns)
                        in_prior_list.append(1 if pt in prior_set_fast else 0)
                    
                    iters_df["in_prior_set"] = in_prior_list
                    
                    streak = 0
                    for v in in_prior_list:
                        if v == 1: streak += 1
                        else: break
                    
                    print(f"    - n={current_n}, beta={beta}: Target in {summary['iterations_to_target']} iters | Streak: {streak}")

if __name__ == "__main__":
    main()