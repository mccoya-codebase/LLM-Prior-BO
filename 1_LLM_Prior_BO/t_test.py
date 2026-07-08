import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_yield_data(priors_dir: Path):
    """
    Loads all set_*.csv files in the directory.
    Returns:
      - set_means: An array of the average yield for each of the 30 sets.
      - all_yields: A flat array of every single individual yield (for distribution plotting).
    """
    set_means = []
    all_yields = []
    
    if not priors_dir.exists():
        fallback_dir = priors_dir.parent / "n_prior_10"
        if fallback_dir.exists():
            priors_dir = fallback_dir
        else:
            print(f"[Error] Directory not found: {priors_dir} or {fallback_dir}")
            return np.array([]), np.array([])
        
    for csv_path in priors_dir.glob("set_*.csv"):
        try:
            df = pd.read_csv(csv_path)
            col = "yield" if "yield" in df.columns else "Yield"
            if col in df.columns:
                yields = pd.to_numeric(df[col], errors='coerce').dropna().values
                if len(yields) > 0:
                    set_means.append(np.mean(yields))
                    all_yields.extend(yields)
        except Exception:
            pass
            
    return np.array(set_means), np.array(all_yields)

def main():
    project_root = Path(r"C:\Users\mccoysa\Projects\LLMPiBO\External_LLMpiBO_code_base")
    priors_base = project_root / "1_LLM_Prior_BO" / "priors"
    
    # Map the dataset folder names to the formal reaction titles for the plot
    reaction_names = {
        "Buchwald": "Buchwald-Hartwig",
        "Suzuki": "Suzuki-Miyaura",
        "Direct": "Direct Arylation"
    }
    
    # Order of panels from left to right
    datasets = ["Buchwald", "Suzuki", "Direct"]
    
    # Initialize the 1-row, 3-column figure
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, ds in enumerate(datasets):
        print(f"\n{'='*50}\nAnalyzing {reaction_names[ds]}\n{'='*50}")
        
        base_dir = priors_base / ds / "gemini_2_5_flash_lite_base" / "n_priors_10"
        informed_dir = priors_base / ds / "gemini_2_5_flash_lite" / "n_priors_10"
        
        base_means, base_raw = load_yield_data(base_dir)
        informed_means, informed_raw = load_yield_data(informed_dir)
        
        if len(base_means) == 0 or len(informed_means) == 0:
            print(f"Skipping {ds} due to missing data.")
            continue

        # --- Statistical Analysis (Welch's t-test) ---
        t_stat, p_value = stats.ttest_ind(base_means, informed_means, equal_var=False)
        
        print(f"Base LLM Mean:     {np.mean(base_means):.2f}% (std: {np.std(base_means):.2f})")
        print(f"Informed LLM Mean: {np.mean(informed_means):.2f}% (std: {np.std(informed_means):.2f})")
        print(f"Mean Difference:   {np.mean(informed_means) - np.mean(base_means):.2f}%")
        print(f"t-statistic:       {t_stat:.4f}")
        print(f"p-value:           {p_value:.4e}")

        # --- Plotting KDE Panel ---
        plot_data_raw = pd.DataFrame({
            "Yield (%)": np.concatenate([base_raw, informed_raw]),
            "Model": ["Base LLM"] * len(base_raw) + ["Paper Informed LLM"] * len(informed_raw)
        })
        
        # Plot to the specific axis (axes[i]) with left limit bounded at 0 via cut=0
        sns.kdeplot(data=plot_data_raw, x="Yield (%)", hue="Model", fill=True, ax=axes[i], palette="tab10", alpha=0.5, cut=0)
        
        # Formatting individual panels with the formal reaction name
        axes[i].set_title(f"{reaction_names[ds]}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Yield (%)", fontsize=12)
        
        # Only show the Y-axis label on the far left plot for cleaner visuals
        if i == 0:
            axes[i].set_ylabel("Density", fontsize=12)
        else:
            axes[i].set_ylabel("")
            
    plt.tight_layout()
    # plt.savefig("Combined_KDE_Distributions.png", dpi=300)
    plt.savefig("Combined_KDE_Distributions.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()