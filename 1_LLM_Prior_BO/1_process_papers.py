"""
Process raw PDF papers for cross-coupling datasets into structured JSON summaries.
Iterates through all PDFs in the Papers directory, dynamically classifying them.
Skips any papers that have already been processed.
Uses Gemini 2.5 Pro (Vertex AI) with strict Pydantic schema enforcement.
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Google GenAI package not installed. Run: pip install google-genai")

# ---------------------------------------------------------
# 1. UNIVERSAL CROSS-COUPLING JSON SCHEMA
# ---------------------------------------------------------
class KeyComponents(BaseModel):
    catalyst: Optional[str] = Field(description="The primary transition metal catalyst/precatalyst used.")
    ligand: Optional[str] = Field(description="The specific ligand(s) evaluated.")
    base: Optional[str] = Field(description="The specific base(s) used in the reaction.")
    additive: Optional[str] = Field(description="Any notable additives used to boost yield or selectivity.")
    electrophile: Optional[str] = Field(description="The electrophile used (e.g., aryl halide or pseudohalide).")
    nucleophile: Optional[str] = Field(description="The nucleophile used (e.g., amine, boronic acid, or C-H substrate).")
    solvent: Optional[str] = Field(description="The optimal solvent used.")

class QuantitativeOutcomes(BaseModel):
    yields_reported: bool = Field(description="Whether specific numerical yields were reported.")
    best_performing_conditions: Optional[str] = Field(description="A brief summary of the exact conditions that produced the highest yields.")
    notes: Optional[str] = Field(description="General notes regarding the quantitative outcomes, scope, or limitations.")

class PaperSummary(BaseModel):
    paper_title: str = Field(description="The title of the paper.")
    reaction_category: str = Field(description="Classify as: 'Buchwald-Hartwig amination', 'Suzuki-Miyaura coupling', 'Direct Arylation', or 'Other/None'.")
    relevance: str = Field(description="Relevance to cross-coupling optimization: 'High', 'Medium', 'Low', or 'None'.")
    key_components: KeyComponents
    mechanistic_insights: List[str] = Field(description="A list of 3-5 critical mechanistic rules, limitations, or trends discovered.")
    quantitative_outcomes: QuantitativeOutcomes

# ---------------------------------------------------------

def main():
    # Initialize the GenAI client via Vertex AI
    client = genai.Client(vertexai=True, project="afrl-il4-rch-polymer-lltl", location="us-central1")
    
    # --- PATH ROUTING BASED ON EXACT DIRECTORY STRUCTURE ---
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    papers_dir = base_dir / "Papers"
    processed_dir = papers_dir / "Processed"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab all PDFs and sort them numerically based on the prefix (e.g., "1_name.pdf")
    all_pdfs = list(papers_dir.glob("*.pdf"))
    
    def sort_key(filepath):
        prefix = filepath.name.split("_")[0]
        return int(prefix) if prefix.isdigit() else 9999
        
    all_pdfs.sort(key=sort_key)

    if not all_pdfs:
        print(f"No PDFs found in {papers_dir}. Please add files and re-run.")
        return
        
    print(f"--- Starting Universal Document Processing ({len(all_pdfs)} files found) ---")

    system_prompt = """
    You are an expert synthetic chemist compiling a structured database for future reference.
    Carefully read the provided research paper and extract the critical reaction information.
    
    CRITICAL INSTRUCTIONS:
    1. Accurately classify the paper into one of the core reaction categories: Buchwald-Hartwig amination, Suzuki-Miyaura coupling, Direct Arylation, or Other.
    2. Pay extremely close attention to the specific ligands, bases, solvents, and additives evaluated.
    3. If the paper is NOT related to synthetic chemistry or cross-coupling (e.g., an analytical chemistry or purely computational methods paper), mark relevance as 'None' and leave irrelevant fields null.
    4. Focus heavily on extracting explicit trends and mechanistic rules that could guide future Bayesian Optimization of these reactions.
    """

    for pdf_path in all_pdfs:
        output_file = processed_dir / f"{pdf_path.stem}.json"
        
        # Safe resume logic: Skip if already processed
        if output_file.exists():
            print(f"  [Skip] Already processed: {pdf_path.name}")
            continue
            
        print(f"Processing: {pdf_path.name}...")
        
        try:
            # Vertex AI Solution: Read the PDF as raw bytes and pass it inline
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
                
            pdf_part = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf"
            )
            
            # Request the structured JSON extraction
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[pdf_part, system_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PaperSummary,
                    temperature=0.1, 
                ),
            )
            
            parsed_json = json.loads(response.text)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4)
                
            classified_type = parsed_json.get("reaction_category", "Unknown")
            print(f"  -> Successfully saved ({classified_type})")
            
        except Exception as e:
            print(f"  -> [ERROR] Failed to process {pdf_path.name}: {e}")

    print("\nAll papers processed successfully.")

if __name__ == "__main__":
    main()