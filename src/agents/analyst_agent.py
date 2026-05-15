"""
analyst_agent.py — Final Verdict Agent (V7: Strict Binary)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict

from src.agents.base_agent import BaseAgent


class AnalystOutput(BaseModel):
    verdict:     str = Field(..., description="Must be exactly 'True' or 'Fake'")
    explanation: str = Field(..., description="A concise 2-sentence explanation summarizing WHY the image supports or contradicts the caption.")

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: str) -> str:
        """
        Normalize LLM verdict to exactly 'True' or 'Fake'.
        OOC cases are treated as Fake.
        """
        if not isinstance(v, str):
            return "Fake" # Fail-safe
        
        mapping = {
            "real":           "True",
            "true":           "True",
            "authentic":      "True",
            "fake_ooc":       "Fake",
            "fake ooc":       "Fake",
            "out-of-context": "Fake",
            "out_of_context": "Fake",
            "ooc":            "Fake",
            "fake":           "Fake",
            "false":          "Fake",
        }
        normalized = mapping.get(v.strip().lower())
        
        if normalized is None:
            print(f"[Analyst] WARNING: Unrecognized verdict '{v}'. Defaulting to 'Fake'.")
            return "Fake"
        
        return normalized


class AnalystAgent(BaseAgent):

    def run(self, deep_analysis: str) -> Dict:
        print("[Analyst] Reading Detective's Report and generating strict binary verdict...")

        system_prompt = """You are the Chief Fact-Checking Judge.
Your sole duty is to read a 4-step visual forensics report provided by your Detective agent and extract the final verdict.
You must output a STRICT BINARY verdict: 'True' or 'Fake'.

RULES:
1. If Detective's STEP 4 says [TRUE] -> verdict = "True"
2. If Detective's STEP 4 says [OUT-OF-CONTEXT] or [FAKE] -> verdict = "Fake" (Misrepresenting context is a form of Fake news).
3. The 'explanation' must be a concise summary (max 2 sentences) of the Detective's logic. Explain WHY the image supports or contradicts the caption. No markdown.

FEW-SHOT EXAMPLES:

EXAMPLE 1
Detective report summary:
  ... STEP 4: [TRUE] — The core visual action matches the caption...
-> {"verdict": "True", "explanation": "The visual evidence supports the core action described in the caption. Conflicts raised by external evidence relate to invisible context and do not contradict the physical events depicted."}

EXAMPLE 2
Detective report summary:
  ... STEP 4: [OUT-OF-CONTEXT] — The core physical action in the image is completely different...
-> {"verdict": "Fake", "explanation": "The image is being used out of context. The visual evidence shows authorities forcefully escorting individuals, which explicitly contradicts the caption's claim of volunteers delivering food."}
"""

        user_prompt = f"""Detective investigation report:
---
{deep_analysis}
---

Task:
Read the report, find the STEP 4 Final Verdict, and output the JSON format.
Return ONLY a JSON object (no markdown, no extra text):
{{"verdict": "True" or "Fake", "explanation": "..."}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # Use Pydantic to strictly enforce the schema
        result = self.llm.chat_completion(messages, response_model=AnalystOutput)

        # Fallback handling
        if not hasattr(result, "verdict") or result.verdict is None:
            print("[Analyst] CRITICAL ERROR: Structured parse failed.")
            return {
                "verdict":     "Fake", # Fail-safe to Fake
                "explanation": "SYSTEM ERROR: Parsing failed. Result defaulted to Fake for safety.",
            }

        return {
            "verdict":     result.verdict,
            "explanation": result.explanation,
        }