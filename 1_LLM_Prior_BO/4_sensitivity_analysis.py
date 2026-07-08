"""
Sensitivity Analysis for LLM Prior Weights.
Automatically loops over Buchwald, Suzuki, and Direct Arylation datasets.
Pulls pre-calculated predicted ranks directly from the centralized predicted_ranks directory
and evaluates the weighted priors against the centralized ground truth data.
Reports Batch Mean, Std, and Max Yields, and outputs comparative boxplots.
Uniform scheme is forced to be the darkest blue.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# DYNAMIC DATASET & WEIGHT SCHEME CONFIGURATIONS
# ---------------------------------------------------------
DATASET_CONFIGS = {
    "Buchwald": {
        "title": "Buchwald-Hartwig",
        "cols": ["additive", "aryl_halide", "base", "ligand"],
        "schemes": {
            "Uniform":         {"additive": 1.0, "aryl_halide": 1.0, "base": 1.0, "ligand": 1.0},
            "Heavy Ligand":    {"additive": 1.0, "aryl_halide": 1.0, "base": 1.0, "ligand": 3.0},
            "Heavy Base":      {"additive": 1.0, "aryl_halide": 1.0, "base": 3.0, "ligand": 1.0},
            "Ignore Additive": {"additive": 0.0, "aryl_halide": 1.0, "base": 1.0, "ligand": 1.0},
            "Ignore Base":     {"additive": 1.0, "aryl_halide": 1.0, "base": 0.0, "ligand": 1.0},
        }
    },
    "Suzuki": {
        "title": "Suzuki-Miyaura",
        "cols": ["Electrophile_SMILES", "Nucleophile_SMILES", "Ligand_SMILES", "Base_SMILES", "Solvent_SMILES"],
        "schemes": {
            "Uniform":         {"Electrophile_SMILES": 1.0, "Nucleophile_SMILES": 1.0, "Ligand_SMILES": 1.0, "Base_SMILES": 1.0, "Solvent_SMILES": 1.0},
            "Heavy Ligand":    {"Electrophile_SMILES": 1.0, "Nucleophile_SMILES": 1.0, "Ligand_SMILES": 3.0, "Base_SMILES": 1.0, "Solvent_SMILES": 1.0},
            "Ignore Solvent":  {"Electrophile_SMILES": 1.0, "Nucleophile_SMILES": 1.0, "Ligand_SMILES": 1.0, "Base_SMILES": 1.0, "Solvent_SMILES": 0.0},
            "Heavy Base":      {"Electrophile_SMILES": 1.0, "Nucleophile_SMILES": 1.0, "Ligand_SMILES": 1.0, "Base_SMILES": 3.0, "Solvent_SMILES": 1.0},
        }
    },
    "Direct": {
        "title": "Direct Arylation",
        "cols": ["Base_SMILES", "Ligand_SMILES", "Solvent_SMILES"],
        "schemes": {
            "Uniform":         {"Base_SMILES": 1.0, "Ligand_SMILES": 1.0, "Solvent_SMILES": 1.0},
            "Heavy Base":      {"Base_SMILES": 3.0, "Ligand_SMILES": 1.0, "Solvent_SMILES": 1.0},
            "Heavy Ligand":    {"Base_SMILES": 1.0, "Ligand_SMILES": 3.0, "Solvent_SMILES": 1.0},
            "Ignore Solvent":  {"Base_SMILES": 1.0, "Ligand_SMILES": 1.0, "Solvent_SMILES": 0.0},
        }
    }
}

# Hardcoded palette to guarantee Uniform is the darkest blue
CUSTOM_PALETTE = {
    "Uniform": "#08519c",         # Darkest blue
    "Heavy Ligand": "#3182bd",    # Dark-medium blue
    "Heavy Base": "#6baed6",      # Medium blue
    "Ignore Additive": "#9ecae1", # Light blue
    "Ignore Solvent": "#9ecae1",  # Light blue (safe to overlap, different plot)
    "Ignore Base": "#c6dbef"      # Lightest blue
}

def load_scores_from_disk(predicted_ranks_dir: Path, mappings: dict, param_columns: list) -> dict:
    value_to_score = {}
    for col in param_columns:
        path = predicted_ranks_dir / f"{col}_scores.json"
        if not path.exists():
            raise FileNotFoundError(f"Score file not found: {path}")
        
        with open(path) as f:
            scored_data = json.load(f)
            
        id_to_value = mappings[col]["id_to_value"]
        options = [str(val) for val in id_to_value.values()]
        value_to_score_col = {}
        
        for val, data in scored_data.items():
            val = str(val).strip()
            score = 0.0
            if isinstance(data, dict):
                try:
                    score = float(data.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
            value_to_score_col[val] = score
            
        for opt in options:
            if opt not in value_to_score_col:
                value_to_score_col[opt] = 0.0
                
        value_to_score[col] = value_to_score_col
        
    return value_to_score

def score_combinations(mappings: dict, value_to_score: dict, df: pd.DataFrame, weights: dict, param_columns: list) -> list:
    unique_combos = df[param_columns].drop_duplicates()
    scored = []
    
    for _, row in unique_combos.iterrows():
        combo = tuple(int(float(row[col])) for col in param_columns)
        
        total = 0.0
        for dim_idx, id_val in enumerate(combo):
            col = param_columns[dim_idx]
            val_str = mappings[col]["id_to_value"].get(str(id_val))
            
            raw_score = 0.0 if val_str is None else value_to_score[col].get(val_str, 0.0)
            total += raw_score * weights[col]
            
        scored.append((total, combo))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def main():
    # ---------------------------------------------------------
    # TOGGLE CONFIGURATIONS HERE
    # ---------------------------------------------------------
    MODEL_NAME = "gemini_2_5_flash_lite"    
    N_PRIORS = 10
    MAX_SETS = 30
    
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent 
    
    all_results = []

    # Iterate through all 3 datasets automatically
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\n=======================================================")
        print(f"--- Processing {dataset_name} ({MODEL_NAME}) ---")
        print(f"=======================================================")
        
        param_columns = config["cols"]
        weight_schemes = config["schemes"]

        data_dir = base_dir / "Data" / dataset_name
        mappings_path = data_dir / "mappings.json"
        dataset_path = data_dir / "processed.csv"
        predicted_ranks_base = script_dir / "predicted_ranks" / MODEL_NAME / dataset_name

        with open(mappings_path) as f:
            mappings = json.load(f)

        df_truth = pd.read_csv(dataset_path)
        for col in param_columns:
            df_truth[col] = df_truth[col].astype(int)
        df_truth["yield"] = pd.to_numeric(df_truth["yield"], errors="coerce").fillna(0.0)

        for set_index in range(MAX_SETS):
            predicted_ranks_dir = predicted_ranks_base / f"rank_{set_index}"
            
            if not predicted_ranks_dir.exists():
                continue
                
            value_to_score = load_scores_from_disk(predicted_ranks_dir, mappings, param_columns)
            
            for scheme_name, scheme_weights in weight_schemes.items():
                scored_combos = score_combinations(mappings, value_to_score, df_truth, scheme_weights, param_columns)
                top_n = [combo for _, combo in scored_combos[:N_PRIORS]]
                
                df_priors = pd.DataFrame(top_n, columns=param_columns)
                merged = pd.merge(df_priors, df_truth, on=param_columns, how='left')
                
                if not merged["yield"].dropna().empty:
                    batch_mean = merged["yield"].mean()
                    batch_std = merged["yield"].std()
                    batch_max = merged["yield"].max()
                    
                    all_results.append({
                        "Dataset": dataset_name,
                        "Set Index": set_index,
                        "Weight Scheme": scheme_name,
                        "Batch Mean Yield": batch_mean,
                        "Batch Std Yield": batch_std,
                        "Batch Max Yield": batch_max
                    })

    if not all_results:
        print("\nNo results found. Double check your predicted_ranks paths.")
        return

    # Compile and save the combined results
    df_results = pd.DataFrame(all_results)
    out_csv = script_dir / f"master_{MODEL_NAME}_weight_sensitivity_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved raw data for all datasets to: {out_csv.name}")

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    print("Generating Boxplots...")
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--"})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    datasets = ["Buchwald", "Suzuki", "Direct"]

    for ax, ds_name in zip(axes, datasets):
        df_ds = df_results[df_results["Dataset"] == ds_name]
        
        if df_ds.empty:
            ax.set_title(f"{DATASET_CONFIGS[ds_name]['title']}\n(No Data)", fontsize=16)
            continue
            
        # Extract the correct order from the config to keep "Uniform" first
        scheme_order = list(DATASET_CONFIGS[ds_name]["schemes"].keys())
            
        sns.boxplot(
            data=df_ds, 
            x="Weight Scheme", 
            y="Batch Mean Yield",
            order=scheme_order,
            ax=ax,
            palette=CUSTOM_PALETTE,
            showmeans=False,
            showfliers=False,
            boxprops={"edgecolor": "#555555", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 1.2},
            capprops={"color": "#555555", "linewidth": 1.2},
            medianprops={"color": "#333333", "linewidth": 1.5}
        )
        
        ax.set_title(DATASET_CONFIGS[ds_name]['title'], fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel("Batch Mean Yield (%)", fontsize=14)
        else:
            ax.set_ylabel("")
            
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(-5, 105)
        sns.despine(ax=ax)

    plt.tight_layout()
    out_plot = script_dir / f"master_{MODEL_NAME}_weight_sensitivity_boxplots.pdf"
    plt.savefig(out_plot, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Plot saved successfully to: {out_plot.name}")

if __name__ == "__main__":
    main()