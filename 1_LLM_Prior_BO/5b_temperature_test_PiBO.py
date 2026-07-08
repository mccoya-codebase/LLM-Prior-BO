"""
Run Prior-guided Bayesian Optimization (PiBO) for Temperature Ablations.
Loops through the isolated temp_test prior directories (T=0.3, T=0.5), 
loads the datasets, and executes the PiBO campaign. 
Outputs are saved to a dedicated temp_test/piBO_runs folder.
"""

import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from botorch.exceptions.warnings import NumericsWarning

# ---------------------------------------------------------
# PATH INJECTION FOR PiBO
# ---------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_base_dir = _script_dir.parent

if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

try:
    from PiBO import PiBOCampaignRunner
except ModuleNotFoundError:
    from PiBO.PiBO import PiBOCampaignRunner

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
    # RUN CONFIGURATION
    # ---------------------------------------------------------
    DATASETS = ["Direct"]
    TEMPS = ["0.3", "0.5"]
    N_PRIORS = [10]               # Focused on N=10 for temp tests
    MAX_SETS = 30
    BETAS = [1]
    PERCENTILE = 2.0              # Target top 2% of yield
    MAX_ITERATIONS = 200          # Safety limit for iterations
    
    # Isolate routes so it doesn't touch the main paper runs
    temp_priors_base = _script_dir / "temp_test" / "priors"
    temp_runs_base = _script_dir / "temp_test" / "piBO_runs"

    print(f"--- Starting PiBO Temperature Test Campaigns ---")
    print(f"Priors Source: {temp_priors_base}")
    print(f"Output Target: {temp_runs_base}\n")

    for dataset_name in DATASETS:
        print(f"===============================================================")
        print(f"--- Processing {dataset_name} ---")
        
        config = DATASET_CONFIGS[dataset_name]
        param_columns = config["param_columns"]

        # 1. Load Ground Truth Data
        data_dir = _base_dir / "Data" / dataset_name
        dataset_path = data_dir / "processed.csv"
        
        if not dataset_path.exists():
            print(f"[Error] Ground truth not found: {dataset_path}")
            continue

        df = pd.read_csv(dataset_path)
        df["yield"] = pd.to_numeric(df["yield"], errors="coerce").fillna(0.0)

        # Standardize parameter values to strings for BayBE using the dynamic columns
        param_values = []
        for c in param_columns:
            df[c] = df[c].apply(lambda x: str(int(float(x))) if pd.notna(x) else x)
            valid_strs = sorted(list(df[c].dropna().unique()), key=int)
            param_values.append(valid_strs)
        
        # Calculate global target yield threshold
        target_yield = float(np.percentile(df["yield"].values, 100 - PERCENTILE))
        print(f"Target yield (top {PERCENTILE}%) for {dataset_name}: {target_yield:.2f}\n")

        # Initialize Runner exactly matching the original script
        runner = PiBOCampaignRunner(
            param_names=param_columns,
            param_values=param_values,
            df=df,
            yield_col="yield",
        )
        
        for temp in TEMPS:
            llm_model = f"gemini_2_5_flash_lite_temp_{temp}"
            print(f"\n  >> Active Model: {llm_model}")
            
            for current_n in N_PRIORS:
                for set_num in range(MAX_SETS):
                    
                    prior_path = temp_priors_base / dataset_name / llm_model / f"n_priors_{current_n}" / f"set_{set_num}.csv"
                    if not prior_path.exists():
                        continue

                    # Load the prior set using the exact same loader function
                    raw_priors = load_prior_set(prior_path, len(param_columns))
                    
                    prior_set = []
                    for p in raw_priors:
                        if all(str(int(p[i])) in param_values[i] for i in range(len(p))):
                            prior_set.append(p)
                    
                    prior_set_fast = set(prior_set)

                    out_base = temp_runs_base / dataset_name / llm_model / f"n_priors_{current_n}" / f"set_{set_num}"
                    out_base.mkdir(parents=True, exist_ok=True)

                    for beta in BETAS:
                        results_dir = out_base / f"beta_{beta}"
                        results_dir.mkdir(parents=True, exist_ok=True)
                        
                        iters_csv = results_dir / "iterations.csv"
                        if iters_csv.exists():
                            # Don't run if it already completed
                            continue

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
                        
                        # Apply standard PiBO tracker columns exactly as in your original script
                        in_prior_list = []
                        for _, row in iters_df.iterrows():
                            pt = tuple(int(float(row[col])) for col in param_columns)
                            in_prior_list.append(1 if pt in prior_set_fast else 0)
                        
                        iters_df["in_prior_set"] = in_prior_list
                        iters_df.to_csv(iters_csv, index=False)
                        
                        streak = 0
                        for v in in_prior_list:
                            if v == 1: streak += 1
                            else: break
                        
                        print(f"    - Set {set_num}, Beta {beta}: Target in {summary['iterations_to_target']} iters | Streak: {streak}")

if __name__ == "__main__":
    main()