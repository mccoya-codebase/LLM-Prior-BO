"""
Base class for LLM clients.
"""

import json
from abc import ABC, abstractmethod
from typing import List, Dict


def parse_json_recommendations(response_text: str, provider: str = "LLM") -> list:
    """Parse JSON array from LLM response"""
    try:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        recommendations = json.loads(text.strip())
        return recommendations if isinstance(recommendations, list) else []
    except json.JSONDecodeError as e:
        print(f"Failed to parse {provider} response as JSON: {e}")
        print(f"Response was: {response_text[:500]}...")
        return []


class BaseLLMClient(ABC):

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the LLM client.
        """
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def extract_recommendations(
        self,
        paper_text: str,
        reaction_type: str,
        available_components: Dict[str, List[str]],
    ) -> List[Dict]:
        """
        Extract reaction recommendations from paper text.
        """
        pass

    def generate_prior_recommendations(
        self,
        paper_text: str,
        reaction_type: str,
        available_components: Dict[str, List[str]],
        n_recommendations: int = 10,
    ) -> List[Dict]:
        """
        Generate prior recommendations for BO from paper analysis.
        """
        recommendations = self.extract_recommendations(
            paper_text=paper_text,
            reaction_type=reaction_type,
            available_components=available_components,
        )
        return recommendations[:n_recommendations]

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass
