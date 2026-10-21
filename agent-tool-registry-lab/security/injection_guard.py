import re
from typing import Any, Tuple

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"(?i)system\s*:\s*you\s+are\s+now",
    r"(?i)disregard\s+(the\s+)?system\s+prompt",
    r"(?i)new\s+instructions\s*:",
    r"(?i)bypass\s+security\s+checks",
]


class PromptInjectionDefense:
    @staticmethod
    def inspect_untrusted_content(text: str) -> Tuple[bool, str]:
        """
        Scans external raw tool outputs for classic indirect prompt injection attacks.
        Returns: (is_suspicious, reason)
        """
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text):
                return (
                    True,
                    f"Triggered suspicious indirect injection heuristic: '{pattern}'",
                )
        return False, "Clean"

    @staticmethod
    def sanitize_and_encapsulate(content: Any, tool_name: str) -> str:
        """
        Wraps content in non-standard semantic boundaries so LLM system prompts
        can treat output strictly as passive data rather than instructions.
        """
        raw_str = str(content)
        # Neutralize common markdown/injection breakout sequences
        neutralized = (
            raw_str.replace("```", "'''")
            .replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
        )
        return (
            f"<untrusted_tool_data source='{tool_name}'>\n"
            f"{neutralized}\n"
            f"</untrusted_tool_data>"
        )
