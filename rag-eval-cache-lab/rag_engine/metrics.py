"""
RAG Evaluation Metrics (Context Precision & Faithfulness Mechanics).
"""

import re
from typing import List


class RAGEvaluator:
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    @staticmethod
    def context_precision(
        retrieved_contexts: List[str], ground_truth: str, threshold: float = 0.25
    ) -> float:
        """
        Calculates Context Precision (Mean Average Precision style).
        Precision@k = (Relevance Indicator @ k) * (Relevant chunks up to k / k)
        Context Precision = Sum(Precision@k * I_k) / Total Relevant Chunks
        """
        gt_tokens = set(re.findall(r"\w+", ground_truth.lower()))
        relevance_indicators: List[int] = []

        # Simple lexical overlap / jaccard heuristic to simulate LLM Relevance Judge
        for ctx in retrieved_contexts:
            ctx_tokens = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(gt_tokens.intersection(ctx_tokens))
            precision_score = overlap / max(1, len(gt_tokens))
            relevance_indicators.append(1 if precision_score >= threshold else 0)

        total_relevant = sum(relevance_indicators)
        if total_relevant == 0:
            return 0.0

        precisions_at_k = []
        running_relevant = 0

        for k, is_relevant in enumerate(relevance_indicators, start=1):
            if is_relevant == 1:
                running_relevant += 1
                precisions_at_k.append(running_relevant / k)

        return float(sum(precisions_at_k) / total_relevant)

    @staticmethod
    def faithfulness(
        generated_answer: str,
        retrieved_contexts: List[str],
        overlap_threshold: float = 0.5,
    ) -> float:
        """
        Faithfulness = Number of statements supported by context / Total statements.
        Mimics Ragas faithfulness without calling external models.
        """
        statements = RAGEvaluator._split_into_sentences(generated_answer)
        if not statements:
            return 0.0

        all_context = " ".join(retrieved_contexts).lower()
        context_tokens = set(re.findall(r"\w+", all_context))

        supported_statements = 0

        for stmt in statements:
            stmt_tokens = set(re.findall(r"\w+", stmt.lower()))
            # Remove punctuation/stop words heuristic
            informative_tokens = {t for t in stmt_tokens if len(t) > 2}
            if not informative_tokens:
                continue

            matched = len(informative_tokens.intersection(context_tokens))
            support_ratio = matched / len(informative_tokens)

            # If token support ratio exceeds threshold, statement is faithful/grounded
            if support_ratio >= overlap_threshold:
                supported_statements += 1

        return float(supported_statements / len(statements))
