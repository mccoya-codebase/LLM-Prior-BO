import os
from typing import List, Dict
from .base_client import BaseLLMClient, parse_json_recommendations

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

class GeminiClient(BaseLLMClient):
    """Google Gemini API client via Vertex AI for generating reaction recommendations."""

    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(self, api_key: str = None, model: str = None):
        # We ignore api_key here since we are using Vertex IAM auth
        super().__init__(api_key=None, model=model)

        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google GenAI package not installed. "
                "Run: pip install google-genai"
            )

        self.model = model or self.DEFAULT_MODEL
        
        # Initialize the new GenAI client specifically for Vertex AI
        self.client = genai.Client(
            vertexai=True, 
            project="afrl-il4-rch-polymer-lltl", 
            location="us-central1"
        )

    @property
    def provider_name(self) -> str:
        return "Gemini (Vertex AI)"

    def generate_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        """Generate a completion using Gemini API via Vertex in strict JSON mode."""
        
        # Configure the call using the new SDK types
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        }
        
        # Pass system prompt properly into the config if provided
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
            
        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
            
        return response.text

    def extract_recommendations(
        self,
        paper_text: str,
        reaction_type: str,
        available_components: Dict[str, List[str]],
    ) -> List[Dict]:
        """
        Extract reaction recommendations from text using Gemini.
        """
        component_desc = "\n".join(
            [f"- {name}: {values}" for name, values in available_components.items()]
        )

        prompt = f"""You are an expert chemist specializing in {reaction_type} reactions.
        Your task is to analyze research papers and recommend optimal reaction conditions.

        Available components for this reaction:
        {component_desc}

        You must ONLY recommend combinations using the available components listed above.

        Based on the following research paper, recommend the top 10 reaction conditions
        that are most likely to give high yields for {reaction_type} reactions.

        Paper content:
        {paper_text[:15000]}

        Return a JSON array of recommendations, where each recommendation is an object with keys
        matching the component names. Example format:
        [
            {{"component_name": "value1", ...}},
            ...
        ]

        Only include components from the available list. Return ONLY the JSON array, no other text."""

        response = self.generate_completion(prompt=prompt, temperature=0.3)
        return parse_json_recommendations(response, "Gemini")