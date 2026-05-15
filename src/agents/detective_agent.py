# detective_agent.py — prompt V10: Smart Protocol (Mutually Exclusive)

from typing import Dict, List
from src.agents.base_agent import BaseAgent
from src.llm_provider import llm_provider

class DetectiveAgent(BaseAgent):

    def run(
        self,
        conflicts:   List[str],
        image_url:   str,
        caption:     str,
    ) -> Dict:
        print("[Detective] Gemma 4 E4B visual investigation...")

        if not conflicts:
            return {
                "deep_analysis": (
                    "No conflicts flagged by retrieval. "
                    "Image-caption appears consistent without further visual investigation."
                )
            }

        # Prompt V10: Phân biệt "Tương thích" và "Xung khắc tuyệt đối"
        prompt = f"""You are an elite Visual Fact-Checking Detective. Your job is to verify if the Full Caption is telling the truth about the image, using both your eyes and the External Evidence.

═══════════════════════════════════════════
FULL CAPTION (The claim to verify):
"{caption}"
═══════════════════════════════════════════

EXTERNAL CONFLICTS TO CROSS-CHECK:
{chr(10).join(f"  [{i+1}] {c}" for i, c in enumerate(conflicts)) if conflicts else "  None. Verify caption against image only."}

═══════════════════════════════════════════
THE 3 RULES OF FACT-CHECKING (SMART PROTOCOL):

1. VISUAL CONTRADICTION: If the physical action in the image blatantly contradicts the caption, rate as [FAKE].
2. MUTUALLY EXCLUSIVE CONTEXT (CRITICAL): You must ONLY rate as [FAKE] if the Evidence proves a MUTUALLY EXCLUSIVE fact about the Date, Location, or Event.
   - Mutually Exclusive (FAKE): Caption says "2022", Evidence says "2020". Caption says "Protest", Evidence says "Quarantine".
   - Compatible (TRUE): Caption says "Self-quarantine", Evidence says "Police enforcing quarantine". (These describe the same event from different angles. Do NOT punish the caption for vocabulary differences).
3. IGNORE META-COMMENTARY & MISSING INFO: 
   - If the evidence says "Unknown", ignore it (Rate [TRUE]).
   - Ignore meta-journalism phrases in the evidence like "A misleading TikTok post claimed..." or "Fact-checkers found...". Focus ONLY on the physical facts of the event.

═══════════════════════════════════════════
STEP 1 — SCENE INVENTORY:
  • Core Action: What is physically happening?
  • Setting/People:

STEP 2 — CAPTION vs. IMAGE (Visual Check):
  ✅ CONFIRMED — [Action matches]
  ❌ CONTRADICTED — [Action does not match]

STEP 3 — CONTEXT CROSS-CHECK (Evidence Check):
  For each numbered conflict:
  → Is it MUTUALLY EXCLUSIVE (triggers Rule 2 = FAKE)? 
  → Or is it COMPATIBLE / META-COMMENTARY / UNKNOWN (triggers Rule 2/3 = TRUE)?

STEP 4 — FINAL VERDICT:
  Choose ONE:
  [TRUE] — Visuals match AND no facts are explicitly mutually exclusive.
  [FAKE] — Visuals contradict OR the Evidence proves a mutually exclusive Event/Date/Location.
"""
        
        report = llm_provider.vision_completion(image_url, prompt)
        print("[Detective] Visual report complete.")

        print("\n" + "═"*70)
        print("🕵️ BÁO CÁO ĐIỀU TRA TỪ DETECTIVE (GEMMA 4 - V10)")
        print("═"*70)
        print(report)
        print("═"*70 + "\n")

        return {"deep_analysis": report}