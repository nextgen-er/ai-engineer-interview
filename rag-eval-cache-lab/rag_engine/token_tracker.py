"""
Token tracking and cost ledger for RAG pipelines.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelCostProfile:
    input_cost_per_million: float  # e.g., $0.150 for GPT-4o-mini
    output_cost_per_million: float  # e.g., $0.600 for GPT-4o-mini


class TokenCostTracker:
    def __init__(self, cost_profile: ModelCostProfile):
        self.cost_profile = cost_profile
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.cached_queries_saved: int = 0
        self.saved_input_tokens: int = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Lightweight fallback token estimator (~0.75 words per token or 4 chars/token).
        For tiktoken/exact BPE, replace with tiktoken.encoding_for_model().
        """
        if not text:
            return 0
        words = text.strip().split()
        return max(1, int(len(words) * 1.33))

    def record_transaction(
        self, prompt: str, completion: str, is_cache_hit: bool = False
    ) -> Dict[str, Any]:
        prompt_tokens = self.estimate_tokens(prompt)
        completion_tokens = self.estimate_tokens(completion)

        if is_cache_hit:
            self.cached_queries_saved += 1
            self.saved_input_tokens += prompt_tokens
            cost = 0.0
            return {
                "cache_hit": True,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": 0.0,
            }

        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens

        prompt_cost = (
            prompt_tokens / 1_000_000
        ) * self.cost_profile.input_cost_per_million
        completion_cost = (
            completion_tokens / 1_000_000
        ) * self.cost_profile.output_cost_per_million
        cost = prompt_cost + completion_cost

        return {
            "cache_hit": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }

    def summary(self) -> Dict[str, Any]:
        total_cost = (
            self.total_input_tokens / 1_000_000
        ) * self.cost_profile.input_cost_per_million + (
            self.total_output_tokens / 1_000_000
        ) * self.cost_profile.output_cost_per_million
        saved_cost = (
            self.saved_input_tokens / 1_000_000
        ) * self.cost_profile.input_cost_per_million

        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(total_cost, 6),
            "cache_hits": self.cached_queries_saved,
            "estimated_saved_cost_usd": round(saved_cost, 6),
        }
