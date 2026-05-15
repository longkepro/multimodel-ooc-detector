"""
contextual_items_extractor.py — Stage 1b: 2-Dimension Contextual Extraction (Date & Location)

Thay vì cố gắng trích xuất 6 thuộc tính dễ gây nén mất mát dữ liệu (Lossy Compression), 
bản này chỉ trích xuất 2 thuộc tính minh bạch nhất: Date và Location để làm Gợi ý (Hints).
Toàn bộ phần đánh giá mâu thuẫn phức tạp sẽ nhường lại cho LLM xử lý dựa trên Raw Text.
"""

from pydantic import BaseModel, Field
from src.llm_provider import llm_provider

# ──────────────────────────────────────────────────────────────
# INTERMEDIATE: 2 Contextual Items (Hints)
# ──────────────────────────────────────────────────────────────

class ContextualItems(BaseModel):
    """
    The 2 core context attributes used as hints.
    Each field is a string answer or "Unknown" if evidence is insufficient.
    """
    location:   str = Field(default="Unknown", description="Where was the event taken?")
    date:       str = Field(default="Unknown", description="When was the event taken?")

# ──────────────────────────────────────────────────────────────
# CORE QUESTIONS
# ──────────────────────────────────────────────────────────────

_CONTEXT_QUESTIONS: dict[str, str] = {
    "location":   "Where was the event in this image taken? (City, Country, Specific place)",
    "date":       "When was the event in this image taken? (Year, Month, Time context)",
}

# ──────────────────────────────────────────────────────────────
# EVIDENCE PREPARATION
# ──────────────────────────────────────────────────────────────

def _build_evidence_context(
    evidence_list: list[dict],
    visual_entities: list[str],
) -> str:
    """Combine ranked evidence into a single context string for the LLM."""
    lines = []

    if visual_entities:
        lines.append("## VISUAL ENTITIES")
        for ent in visual_entities:
            lines.append(f"  - {ent}")
        lines.append("")

    lines.append("## RETRIEVED EVIDENCE")
    for i, ev in enumerate(evidence_list):
        lines.append(f"### Source {i+1} ({ev.get('source_type', 'unknown')})")
        lines.append(f"Title: {ev.get('title', '')}")

        captions = ev.get("image_captions", [])
        if captions:
            lines.append("Image captions found on page:")
            for cap in captions[:3]:
                lines.append(f"  \"{cap}\"")

        lines.append(f"Article text:\n{ev.get('text', '')[:600]}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# STEP 1: QA for 2 context items
# ──────────────────────────────────────────────────────────────

def _extract_contextual_items(
    evidence_context: str,
    source_label: str = "evidence",
) -> ContextualItems:
    print(f"[Contextual] Extracting 2 context items (Date/Location) from {source_label}...")

    questions_block = "\n".join(
        f"  {i+1}. {key.upper()}: {q}"
        for i, (key, q) in enumerate(_CONTEXT_QUESTIONS.items())
    )

    system_prompt = (
        "You are an expert Context Extraction AI. Your task is to extract information "
        "from the provided text based on 2 specific dimensions.\n"
        "Rules:\n"
        "- Keep the extracted answers concise but highly informative.\n"
        "- If the text DOES NOT explicitly state the answer, you MUST output 'Unknown'.\n"
        "- Do not guess or hallucinate.\n"
        "Respond ONLY with a JSON object."
    )

    user_prompt = f"""Using the evidence below, answer these 2 questions:

{questions_block}

---
TEXT:
{evidence_context}
---

Return ONLY a JSON object with these exact keys:
{{
  "location": "...",
  "date": "..."
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    result = llm_provider.chat_completion(messages, response_model=ContextualItems)

    if isinstance(result, str):
        import json, re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                result = ContextualItems(**data)
            except Exception:
                result = ContextualItems()
        else:
            result = ContextualItems()

    print(f"[Contextual] Extracted: {result.model_dump()}")
    return result


# ──────────────────────────────────────────────────────────────
# PUBLIC API (SẠCH SẼ, TRẢ VỀ JSON MODEL)
# ──────────────────────────────────────────────────────────────

def extract_contextual_svo(
    evidence_list: list[dict],
    visual_entities: list[str],
    source_label: str = "evidence",
) -> dict:
    """
    Bỏ qua hoàn toàn bước gọi LLM trích xuất Hint.
    Trả về dictionary rỗng để Qwen tự đọc Raw Text.
    """
    print(f"[Contextual] Bỏ qua trích xuất Hints (Disabled to prevent Anchor Bias).")
    return {}