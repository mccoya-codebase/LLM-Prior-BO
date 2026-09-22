import pandas as pd
import json
from pathlib import Path

# This script processes raw experimental data for different reaction types.


# Define the base path relative to the script's location.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "Data"

# --- Dataset Configurations ---
# This dictionary holds the specific settings for each dataset, including filenames,
# column mappings, ID start values, and any special filtering rules.
DATASET_CONFIGS = {
    "Buchwald": {
        "input_file": "experiment_index.csv",
        "input_cols": ['Additive_SMILES', 'Aryl_halide_SMILES', 'Base_SMILES', 'Ligand_SMILES', 'yield'],
        "output_cols": ['additive', 'aryl_halide', 'base', 'ligand', 'yield'],
        "component_cols": ['additive', 'aryl_halide', 'base', 'ligand'],
        "id_start": 1,  # IDs will start from 1
        "filters": None # No special filtering for this dataset
    },
    "Suzuki": {
        "input_file": "experiment_index.csv",
        "input_cols": ['Electrophile_SMILES', 'Nucleophile_SMILES', 'Ligand_SMILES', 'Base_SMILES', 'Solvent_SMILES', 'yield'],
        "output_cols": ['electrophile', 'nucleophile', 'ligand', 'base', 'solvent', 'yield'],
        "component_cols": ['electrophile', 'nucleophile', 'ligand', 'base', 'solvent'],
        "id_start": 1,  # IDs will start from 1
        "filters": None # No special filtering for this dataset
    },
    "Direct": {
        "input_file": "experiment_index.csv",
        "input_cols": ['Ligand_SMILES', 'Base_SMILES', 'Solvent_SMILES', 'yield'],
        "output_cols": ['ligand', 'base', 'solvent', 'yield'],
        "component_cols": ['ligand', 'base', 'solvent'],
        "id_start": 0,  # Special case: IDs start from 0 for Direct Arylation
        "filters": {    # Special case: Filter by these conditions
            "Concentration": 0.1,
            "Temp_C": 105.0
        }
    }
}

def process_single_dataset(dataset_name, config):
    """
    Finds the raw CSV file for a dataset, applies specific processing rules,
    converts SMILES to integer IDs, and saves the output files.
    """
    raw_data_path = DATA_DIR / dataset_name / config["input_file"]
    output_dir = DATA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing Dataset: {dataset_name} ---")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Skipping: Raw data file not found at '{raw_data_path}'\n")
        return

    # Handle 'conversion' column as a fallback for 'yield'
    if "yield" not in df.columns and "conversion" in df.columns:
        df = df.rename(columns={"conversion": "yield"})

    # Validate that all expected input columns exist
    missing_cols = [col for col in config["input_cols"] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in '{raw_data_path}': {missing_cols}\n")
        return

    # --- Special Filtering for Direct Arylation ---
    if config.get("filters"):
        initial_len = len(df)
        for col, value in config["filters"].items():
            if col in df.columns:
                # Ensure correct type for comparison
                df = df[df[col].astype(type(value)) == value]
            else:
                print(f"Warning: Filter column '{col}' not found in '{raw_data_path}'.")

    # --- Data Processing and Mapping ---
    column_map = dict(zip(config["input_cols"], config["output_cols"]))
    
    df_processed = df[config["input_cols"]].copy()
    df_processed.rename(columns=column_map, inplace=True)
    df_processed = df_processed[config["output_cols"]]

    mappings = {}
    id_start = config.get("id_start", 0)  # Default to 0 if not specified
    for col in config["component_cols"]:
        unique_values = sorted(df_processed[col].dropna().unique())
        value_to_id = {value: i + id_start for i, value in enumerate(unique_values)}
        id_to_value = {i + id_start: value for i, value in enumerate(unique_values)}

        mappings[col] = {
            'value_to_id': value_to_id,
            'id_to_value': {str(k): v for k, v in id_to_value.items()}
        }
        df_processed[col] = df_processed[col].map(value_to_id)

    # Save processed files
    processed_csv_path = output_dir / "processed.csv"
    df_processed.to_csv(processed_csv_path, index=False)
    print(f"Successfully generated: '{processed_csv_path}'")

    mappings_json_path = output_dir / "mappings.json"
    with open(mappings_json_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"Successfully generated: '{mappings_json_path}'\n")

def main():
    """
    Main entry point that executes the processing loop sequentially over all configured datasets.
    """
    print(f"Starting data processing from script directory: {SCRIPT_DIR}")
    print(f"Looking for data in base directory: {DATA_DIR}\n")
    
    for dataset_name, config in DATASET_CONFIGS.items():
        process_single_dataset(dataset_name, config)
    
    print("All configured datasets have been checked and processed.")

if __name__ == '__main__':
    main()
