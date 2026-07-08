"""
Generate Synthetic Prior Sets (Ideal, Misleading, etc.) with adaptable indexing.
Outputs route to the new centralized directory structure.
"""

import csv
from pathlib import Path
import pandas as pd

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

def get_starting_index(out_base: Path) -> int:
    """Scan directory to find the highest set_X.csv and return X + 1."""
    if not out_base.exists():
        return 0
    
    existing_indices = []
    for p in out_base.glob("set_*.csv"):
        try:
            idx = int(p.stem.split("_")[1])
            existing_indices.append(idx)
        except ValueError:
            pass
            
    return max(existing_indices) + 1 if existing_indices else 0


def main():
    # ---------------------------------------------------------
    # CONFIGURATION VARIABLES
    # ---------------------------------------------------------
    DATASET_NAME = "Suzuki"  # Toggle: "Buchwald", "Suzuki", or "Direct"
    N_PRIORS = [50,100]   # The prior batch sizes you want to generate
    N_SETS = 30                # Number of sets to generate per prior size
    SEED = None                # Set an integer for deterministic sampling
    # ---------------------------------------------------------

    if DATASET_NAME not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{DATASET_NAME}'. Choose from: {list(DATASET_CONFIGS.keys())}")
        
    config = DATASET_CONFIGS[DATASET_NAME]
    param_columns = config["param_columns"]

    # --- PATH ROUTING ---
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent 
    
    # Input: Pull from central Data folder
    data_path = base_dir / "Data" / DATASET_NAME / "processed.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

    # Output: Route to local priors folder organized by Dataset
    priors_root = script_dir / "priors" / DATASET_NAME

    # --- DATA PROCESSING ---
    df = pd.read_csv(data_path)
    
    # Extract yield safely
    yield_col = "yield" if "yield" in df.columns else "Yield"
    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce").fillna(0.0)
    
    q25 = df[yield_col].quantile(0.25)
    q75 = df[yield_col].quantile(0.75)
    q98 = df[yield_col].quantile(0.98)

    # Define synthetic variant pools
    pools = {
        "misleading": df[df[yield_col] <= q25],
        "ideal": df[(df[yield_col] >= q75) & (df[yield_col] < q98)]
    }

    print(f"--- Generating Priors for {DATASET_NAME} ---")
    
    for n_priors in N_PRIORS:
        for prior_type, pool in pools.items():
            if len(pool) < n_priors:
                print(f"    [Warning] Only {len(pool)} points in '{prior_type}' pool. Sampling with replacement to get {n_priors}.")
            if len(pool) == 0:
                print(f"    [Error] No points found for '{prior_type}' pool! Skipping.")
                continue
                
            out_base = priors_root / prior_type / f"n_priors_{n_priors}"
            out_base.mkdir(parents=True, exist_ok=True)
            
            start_index = get_starting_index(out_base)
            end_index = start_index + N_SETS
            
            for current_index in range(start_index, end_index):
                replace = len(pool) < n_priors
                rng = (SEED + current_index) if SEED is not None else None
                
                sampled = pool.sample(n=n_priors, replace=replace, random_state=rng)
                
                # Extract only the required categorical string parameters
                priors = [tuple(int(float(row[c])) for c in param_columns) for _, row in sampled.iterrows()]
                
                out_path = out_base / f"set_{current_index}.csv"
                with open(out_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(param_columns)
                    writer.writerows(priors)
                    
            print(f" -> Wrote {N_SETS} '{prior_type}' sets (N={n_priors}) | Path: {out_base.relative_to(base_dir)}")

if __name__ == "__main__":
    main()