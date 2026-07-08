"""
Generate Prior Sets via per-dimension scoring.
Toggle between Gemini 2.5 Pro, Gemini 2.5 Flash-Lite, and Llama 3.3 (Vertex MaaS).
Automatically handles Context Caching for supported models and dynamic folder routing.
"""

import csv
import json
import sys
import os
from pathlib import Path
import pandas as pd
import requests

# --- GOOGLE AUTH & GEMINI IMPORTS ---
import google.auth
import google.auth.transport.requests

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Google GenAI package not installed. Run: pip install google-genai")

# ---------------------------------------------------------
# PATH INJECTION FOR LLM
# ---------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_base_dir = _script_dir.parent  

if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))

from LLM import parse_json_recommendations

# ---------------------------------------------------------
# DATASET & MODEL CONFIGURATIONS
# ---------------------------------------------------------
DATASET_CONFIGS = {
    "Buchwald": {
        "param_columns": ["additive", "aryl_halide", "base", "ligand"],
        "reaction_name": "Buchwald-Hartwig amination"
    },
    "Suzuki": {
        "param_columns": ["Electrophile_SMILES", "Nucleophile_SMILES", "Ligand_SMILES", "Base_SMILES", "Solvent_SMILES"],
        "reaction_name": "Suzuki-Miyaura coupling"
    },
    "Direct": {
        "param_columns": ["Base_SMILES", "Ligand_SMILES", "Solvent_SMILES"],
        "reaction_name": "Direct Arylation"
    }
}

MODEL_CONFIGS = {
    "gemini-2.5-pro": {
        "api_model": "gemini-2.5-pro",
        "dir_name": "gemini_2_5_pro",
        "supports_caching": True,
        "provider": "google_genai"
    },
    "gemini-2.5-flash-lite": {
        "api_model": "gemini-2.5-flash-lite", 
        "dir_name": "gemini_2_5_flash_lite",
        "supports_caching": True,
        "provider": "google_genai"
    },
    "llama-3.3": {
        "api_model": "meta/llama-3.3-70b-instruct-maas",
        "dir_name": "llama_3_3",
        "supports_caching": False,
        "provider": "vertex_maas"
    }
}

# ---------------------------------------------------------
# UNIFIED LLM CLIENT
# ---------------------------------------------------------
class UnifiedLLMClient:
    """Smart router client handling Gemini (with caching) and Llama 3.3 (Vertex MaaS)."""
    def __init__(self, model_choice: str, project_id="afrl-il4-rch-polymer-lltl", location="us-central1"):
        self.config = MODEL_CONFIGS[model_choice]
        self.provider = self.config["provider"]
        self.api_model = self.config["api_model"]
        self.supports_caching = self.config["supports_caching"]
        
        self.project_id = project_id
        self.location = location
        
        self.cached_content = None
        self.system_instruction = "You are a strict data-formatting API. Output only valid JSON arrays."
        
        if self.provider == "google_genai":
            self.client = genai.Client(
                vertexai=True, 
                project=self.project_id, 
                location=self.location
            )
        elif self.provider == "vertex_maas":
            self.url = f"https://{self.location}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi/chat/completions"
            self.creds, _ = google.auth.default()
            self.auth_req = google.auth.transport.requests.Request()

    def create_cache(self, context_text: str):
        """Only fires if the model supports Vertex Caching."""
        if not self.supports_caching:
            return None
            
        print(f"  [Cache] Uploading context to Vertex AI for {self.api_model}...")
        self.cached_content = self.client.caches.create(
            model=self.api_model,
            config=types.CreateCachedContentConfig(
                system_instruction=self.system_instruction,
                contents=[context_text],
            )
        )
        print(f"  [Cache] Created successfully: {self.cached_content.name}")
        return self.cached_content.name
        
    def delete_cache(self):
        if self.cached_content and self.supports_caching:
            print(f"  [Cache] Deleting {self.cached_content.name}...")
            self.client.caches.delete(name=self.cached_content.name)
            self.cached_content = None

    def generate_completion(self, prompt: str, temperature: float = 0.8, max_tokens: int = 8192) -> str:
        if self.provider == "google_genai":
            config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            }
            if self.cached_content:
                config_kwargs["cached_content"] = self.cached_content.name
            else:
                config_kwargs["system_instruction"] = self.system_instruction
                
            config = types.GenerateContentConfig(**config_kwargs)
            response = self.client.models.generate_content(
                model=self.api_model,
                contents=prompt,
                config=config
            )
            return response.text
            
        elif self.provider == "vertex_maas":
            self.creds.refresh(self.auth_req)
            headers = {
                "Authorization": f"Bearer {self.creds.token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.api_model,
                "messages": [
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(self.url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"Llama API Error {response.status_code}: {response.text}")

# ---------------------------------------------------------

def load_processed_summaries(processed_dir: Path) -> str:
    if not processed_dir.exists():
        raise FileNotFoundError(f"Mixed Sets not found at: {processed_dir}")
    json_files = sorted(processed_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {processed_dir}.")
    texts = [p.read_text(encoding='utf-8').strip() for p in json_files]
    return "[\n" + ",\n".join(texts) + "\n]"

def get_score_prompt(dimension_name: str, options: list, reaction_name: str, paper_text: str = None) -> str:
    options_blob = "\n".join(f"- {opt}" for opt in options)
    
    context_section = f"\n\nContext Papers:\n{paper_text}\n" if paper_text else ""
    
    return f"""Based on the provided research paper summaries about {reaction_name} reactions, evaluate these {dimension_name} options for achieving high conversion.{context_section}

Options:
{options_blob}

For each option, provide:
- A score from 0.0 to 1.0 (1.0 being the best)

CRITICAL INSTRUCTIONS:
1. Output EXACTLY and ONLY a valid JSON array of objects.
2. NO conversational text and NO markdown formatting (do not use ```json).
3. Provide ONLY the "value" and "score" keys.

Output Format:
[
  {{"value": "<exact string from options>", "score": 0.9}}
]"""

def fetch_and_save_scores(llm_client, mappings: dict, predicted_ranks_dir: Path, param_columns: list, reaction_name: str, paper_text: str = None) -> dict:
    value_to_score = {}
    
    # Pass context text only if the active model does NOT support caching
    active_context = None if llm_client.supports_caching else paper_text
    
    for col in param_columns:
        id_to_value = mappings[col]["id_to_value"]
        options = [str(val) for val in id_to_value.values()]
        
        prompt = get_score_prompt(col, options, reaction_name, active_context)
        
        response = llm_client.generate_completion(prompt=prompt, temperature=0.3)
        
        try:
            raw = json.loads(response)
        except Exception as e:
            print(f"  [Warning] Native JSON load failed. Attempting fallback parse. Error: {e}")
            raw = parse_json_recommendations(response, llm_client.provider)
            
        scored_data = {}
        value_to_score_col = {}
        
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "value" in item:
                    val = str(item["value"]).strip()
                    try:
                        score = float(item.get("score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    scored_data[val] = {"score": score}
                    value_to_score_col[val] = score

        for opt in options:
            if opt not in value_to_score_col:
                value_to_score_col[opt] = 0.0
                scored_data[opt] = {"score": 0.0}
                
        value_to_score[col] = value_to_score_col
        
        out_path = predicted_ranks_dir / f"{col}_scores.json"
        with open(out_path, "w") as f:
            json.dump(scored_data, f, indent=2)
        print(f"  Saved {out_path.name}")
        
    return value_to_score

def load_scores_from_disk(predicted_ranks_dir: Path, mappings: dict, param_columns: list) -> dict:
    value_to_score = {}
    for col in param_columns:
        path = predicted_ranks_dir / f"{col}_scores.json"
        with open(path) as f:
            scored_data = json.load(f)
            
        id_to_value = mappings[col]["id_to_value"]
        options = [str(val) for val in id_to_value.values()]
        value_to_score_col = {}
        
        for val, data in scored_data.items():
            val = str(val).strip()
            score = float(data.get("score", 0.0)) if isinstance(data, dict) else 0.0
            value_to_score_col[val] = score
            
        for opt in options:
            if opt not in value_to_score_col:
                value_to_score_col[opt] = 0.0
        value_to_score[col] = value_to_score_col
    return value_to_score

def score_combinations(mappings: dict, value_to_score: dict, df: pd.DataFrame, param_columns: list) -> list:
    weights = {col: 1.0 for col in param_columns}
    unique_combos = df[param_columns].drop_duplicates()
    scored = []
    
    for _, row in unique_combos.iterrows():
        combo = tuple(int(float(row[col])) for col in param_columns)
        total = sum(value_to_score[col].get(mappings[col]["id_to_value"].get(str(id_val)), 0.0) * weights[col] 
                    for col, id_val in zip(param_columns, combo))
        scored.append((total, combo))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def get_starting_index(out_base: Path) -> int:
    if not out_base.exists(): return 0
    existing = [int(p.stem.split("_")[1]) for p in out_base.glob("set_*.csv") if p.stem.split("_")[1].isdigit()]
    return max(existing) + 1 if existing else 0

def main():
    # ---------------------------------------------------------
    # CONFIGURATION VARIABLES
    # ---------------------------------------------------------
    DATASET_NAME = "Buchwald"
    
    # TOGGLE MODEL HERE: "gemini-2.5-pro", "gemini-2.5-flash-lite", or "llama-3.3"
    MODEL_CHOICE = "gemini-2.5-pro"
    
    N_PRIORS = [1, 3, 5, 10]
    N_SETS = 30
    SKIP_SCORING = False       
    # ---------------------------------------------------------

    config = DATASET_CONFIGS[DATASET_NAME]
    model_info = MODEL_CONFIGS[MODEL_CHOICE]
    model_dir = model_info["dir_name"]

    data_dir = _base_dir / "Data" / DATASET_NAME
    mappings_path = data_dir / "mappings.json"
    dataset_path = data_dir / "processed.csv"
    processed_dir = _base_dir / "Papers" / "Mixed_Sets" / DATASET_NAME
    
    predicted_ranks_base = _script_dir / "predicted_ranks" / model_dir / DATASET_NAME
    
    with open(mappings_path) as f:
        mappings = json.load(f)
    df = pd.read_csv(dataset_path)

    reference_out_base = _script_dir / "priors" / DATASET_NAME / model_dir / f"n_priors_{N_PRIORS[0]}"
    start_index = get_starting_index(reference_out_base)
    end_index = start_index + N_SETS
    
    print(f"--- Generating {N_SETS} sets via {MODEL_CHOICE} ({DATASET_NAME}) ---")

    llm_client = None

    try:
        if not SKIP_SCORING:
            paper_text = load_processed_summaries(processed_dir)
            llm_client = UnifiedLLMClient(model_choice=MODEL_CHOICE)
            llm_client.create_cache(paper_text)

        for current_index in range(start_index, end_index):
            print(f"Processing set {current_index}...")
            predicted_ranks_dir = predicted_ranks_base / f"rank_{current_index}"
            
            if SKIP_SCORING:
                value_to_score = load_scores_from_disk(predicted_ranks_dir, mappings, config["param_columns"])
            else:
                predicted_ranks_dir.mkdir(parents=True, exist_ok=True)
                value_to_score = fetch_and_save_scores(
                    llm_client, mappings, predicted_ranks_dir, config["param_columns"], config["reaction_name"], paper_text
                )

            scored = score_combinations(mappings, value_to_score, df, config["param_columns"])

            # for n_priors in N_PRIORS:
            #     top_n = [combo for _, combo in scored[:n_priors]]
            #     out_base = _script_dir / "priors" / DATASET_NAME / model_dir / f"n_priors_{n_priors}"
            #     out_base.mkdir(parents=True, exist_ok=True)
                
            #     out_path = out_base / f"set_{current_index}.csv"
            #     with open(out_path, "w", newline="") as f:
            #         writer = csv.writer(f)
            #         writer.writerow(config["param_columns"])
            #         writer.writerows(top_n)

            ## updated so I dont have to go back and do this like I had setup before

            for n_priors in N_PRIORS:
                top_n_combos = [combo for _, combo in scored[:n_priors]]
                
                # Convert choices directly to a small dataframe
                df_prior = pd.DataFrame(top_n_combos, columns=config["param_columns"])
                
                # Map the yield from the ground-truth dataframe loaded at the top of the script
                # (Ensure types match: df[config["param_columns"]] = df[config["param_columns"]].astype(int))
                df_yield_mapped = pd.merge(df_prior, df[config["param_columns"] + ["yield"]], on=config["param_columns"], how='left')
                
                out_base = _script_dir / "priors" / DATASET_NAME / model_dir / f"n_priors_{n_priors}"
                out_base.mkdir(parents=True, exist_ok=True)
                
                out_path = out_base / f"set_{current_index}.csv"
                df_yield_mapped.to_csv(out_path, index=False)
                
        print(f"All {DATASET_NAME} prior sets for {MODEL_CHOICE} generated successfully!")

    finally:
        if llm_client and not SKIP_SCORING:
            llm_client.delete_cache()

if __name__ == "__main__":
    main()