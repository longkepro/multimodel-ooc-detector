"""
utils.py — Multi-Granularity Visual Analysis

Design principles (from EGMMG + EXCLAIM papers):

  EGMMG insight:
    - Extract entities (Subject nodes) from caption FIRST via NER/SVO
    - Use those entities as DYNAMIC CLIP probes — not a fixed label list
    - Reuse SentenceTransformer singleton already in evidence_retriever.py
      (no extra model memory)

  EXCLAIM insight:
    - Multi-granularity analysis: Entity-level -> Event-level -> Scene-level
    - Visual verification must be cross-modal and explainable
    - Output structured enough for DetectiveAgent to reason over directly

Environment split:
  - Local  (Groq): base64 image -> vision LLM -> rich structured description
  - Kaggle (vLLM, text-only): CLIP zero-shot with DYNAMIC labels derived from
    caption entities (EGMMG-style entity graph nodes as visual probes)

Singleton fix:
  - CLIP loaded ONCE via lazy singleton (_get_clip)
  - Never reloaded across calls -> fixes ~600MB reload bug
"""

import re
import base64
import requests
from pathlib import Path
from typing import Union

from src.config import Config
from src.llm_provider import llm_provider


# ──────────────────────────────────────────────────────────────
# SINGLETON: CLIP (loaded ONCE)
# ──────────────────────────────────────────────────────────────

_clip_model = None
_clip_processor = None


def _get_clip() -> tuple:
    """
    Lazy singleton for CLIP.
    Problem fixed: previous code called CLIPModel.from_pretrained() inside the
    function body, reloading 600MB every single call.
    Now: load once on first call, return same objects forever.
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        print("[CLIP] Loading clip-vit-base-patch32 (one-time load, ~600MB)...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print("[CLIP] Ready.")
    return _clip_model, _clip_processor


# ──────────────────────────────────────────────────────────────
# IMAGE LOADING
# ──────────────────────────────────────────────────────────────

def _load_image_bytes(image_source: Union[str, bytes]) -> bytes:
    """Load image from URL, local path, or raw bytes. Raises ValueError on failure."""
    if isinstance(image_source, bytes):
        return image_source

    source = str(image_source)

    if source.startswith(("http://", "https://")):
        try:
            resp = requests.get(
                source,
                allow_redirects=True,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
            )
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            raise ValueError(f"Cannot download image from {source}: {e}") from e
    else:
        path = Path(source)
        if not path.exists():
            raise ValueError(f"Local image file not found: {source}")
        return path.read_bytes()


# ──────────────────────────────────────────────────────────────
# EGMMG: Dynamic entity extraction from caption
# ──────────────────────────────────────────────────────────────

def _extract_caption_entities(caption: str) -> list[str]:
    """
    Extract named entities from caption using spaCy NER.

    EGMMG paper: claim graph nodes = named entities in the caption (PERSON,
    GPE, LOC, ORG, EVENT). These become the dynamic CLIP zero-shot probes
    — replacing the fixed hardcoded label list in the old code.

    Pipeline:
      1. spaCy NER (fast, en_core_web_sm ~12MB)
      2. Fallback: regex capitalized phrases + domain keywords
    """
    # Try spaCy first
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed
            raise ImportError("en_core_web_sm not found")

        doc = nlp(caption)
        priority_types = {"PERSON", "FAC", "ORG", "PRODUCT", "LOC"}
        entities = [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in priority_types and len(ent.text.strip()) > 1
        ]
        if entities:
            # Deduplicate, cap at 6 to keep CLIP batch size manageable
            seen = set()
            deduped = []
            for e in entities:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    deduped.append(e)
            return deduped[:6]

    except (ImportError, Exception):
        pass

    # Fallback: regex for capitalized proper nouns + OOC-relevant event keywords
    proper_nouns = [
        word for word in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', caption)
        if not any(char.isdigit() for char in word)
    ]
    event_keywords = [
        kw for kw in [
            "protest", "election", "military", "police", "flood", "fire",
            "crowd", "rally", "ceremony", "press conference", "summit",
            "arrest", "bombing", "earthquake", "hurricane",
        ]
        if kw.lower() in caption.lower()
    ]

    combined = proper_nouns + event_keywords
    seen = set()
    result = []
    for item in combined:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result[:6]


# ──────────────────────────────────────────────────────────────
# EXCLAIM: Multi-granularity probe labels
# ──────────────────────────────────────────────────────────────

def _build_multigranularity_labels(
    caption_entities: list[str],
) -> dict[str, list[str]]:
    """
    Build CLIP probe labels at 3 granularity levels (EXCLAIM).

    ENTITY-LEVEL: Derived from caption entities (EGMMG claim graph nodes).
                  Each entity gets a positive probe ("a photo of X") and a
                  negative probe ("no X visible") for contrastive scoring.

    EVENT-LEVEL:  Candidate event types relevant to OOC misinformation.
                  Broad enough to cover politics, disasters, sports, etc.

    SCENE-LEVEL:  Basic scene attributes that inform context verification.

    This replaces the hardcoded 30-item list in the old code.
    """
    # ENTITY-LEVEL: dynamic from caption (EGMMG technique)
    entity_probes = []
    for ent in caption_entities:
        entity_probes.append(f"a photo of {ent}")
        entity_probes.append(f"no {ent} visible in this image")

    # EVENT-LEVEL: curated for OOC news misinformation domain
    event_probes = [
        "a political speech or press conference",
        "a protest, march, or demonstration",
        "a military or police operation",
        "a sports event or athletic competition",
        "a natural disaster, flood, or fire",
        "a formal ceremony or state event",
        "a humanitarian or refugee situation",
        "an ordinary civilian daily life scene",
    ]

    # SCENE-LEVEL: spatial and temporal context
    scene_probes = [
        "an outdoor scene in daylight",
        "an outdoor scene at night",
        "an indoor scene",
        "a crowded public space",
        "a government or official building",
        "a conflict or war zone environment",
    ]

    return {
        "entity": entity_probes,
        "event": event_probes,
        "scene": scene_probes,
    }


# ──────────────────────────────────────────────────────────────
# CLIP: Multi-granularity zero-shot classification
# ──────────────────────────────────────────────────────────────

def _clip_multigranularity_analysis(
    image_bytes: bytes,
    caption: str,
) -> str:
    """
    Run CLIP zero-shot classification at 3 granularity levels.

    Core EGMMG technique applied here:
      - Extract entity nodes from caption (claim graph)
      - Use them as CLIP probes to check visual presence
      - Flag entities claimed in caption but not visually confirmed
        (this IS the OOC signal: caption claims X is present, image has no X)

    Returns a structured English report directly usable by DetectiveAgent.
    """
    import torch
    from PIL import Image
    from io import BytesIO

    model, processor = _get_clip()   # singleton — never reloaded

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    caption_entities = _extract_caption_entities(caption)
    label_groups = _build_multigranularity_labels(caption_entities)

    report_lines = ["[CLIP Multi-Granularity Visual Analysis]", ""]

    # ── Per-granularity top-k classification ──
    for granularity, labels in label_groups.items():
        if not labels:
            continue

        inputs = processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,   # CLIP hard limit
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        top_n = min(3, len(labels))
        top_indices = probs.topk(top_n).indices.tolist()

        report_lines.append(f"{granularity.upper()}-LEVEL:")
        for idx in top_indices:
            report_lines.append(f"  - {labels[idx]}  ({probs[idx].item():.1%})")
        report_lines.append("")

    # ── Entity verification: EGMMG cross-graph consistency check ──
    # This is the key OOC signal: does the visual evidence match entity claims?
    if caption_entities:
        report_lines.append("ENTITY CROSS-CHECK (EGMMG claim-evidence graph):")
        report_lines.append(f"  Caption claims these entities: {caption_entities}")
        report_lines.append("  Visual verification:")

        entity_labels = label_groups["entity"]
        if entity_labels:
            inputs = processor(
                text=entity_labels,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

            for ent in caption_entities:
                pos_label = f"a photo of {ent}"
                neg_label = f"no {ent} visible in this image"

                p_pos = probs[entity_labels.index(pos_label)] if pos_label in entity_labels else 0.0
                p_neg = probs[entity_labels.index(neg_label)] if neg_label in entity_labels else 0.0

                if p_pos + p_neg > 0:
                    confidence = p_pos / (p_pos + p_neg)
                    if confidence >= 0.6:
                        verdict = "VISUALLY CONFIRMED"
                    elif confidence >= 0.4:
                        verdict = "UNCERTAIN"
                    else:
                        verdict = "NOT CONFIRMED (possible OOC signal)"

                    report_lines.append(
                        f"    '{ent}': {verdict}  "
                        f"(confirmed={p_pos:.1%} / not-found={p_neg:.1%})"
                    )

    return "\n".join(report_lines)


# ──────────────────────────────────────────────────────────────
# LOCAL PATH: Groq Vision LLM
# ──────────────────────────────────────────────────────────────

def _vision_llm_description(image_bytes: bytes) -> str:
    """
    Multi-granularity visual description via Groq vision LLM.

    Prompt mirrors EXCLAIM's 3 granularity levels so DetectiveAgent
    receives structurally equivalent input regardless of environment.
    All instructions in English only.
    """
    base64_img = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image at three levels of detail. "
                        "Respond in English only. "
                        "State ONLY what is directly observable — no speculation.\n\n"
                        "## ENTITY-LEVEL\n"
                        "List all identifiable entities:\n"
                        "- People: physical appearance, clothing, visible actions\n"
                        "- Organizations: flags, logos, uniforms, signs with text\n"
                        "- Locations: landmarks, geography, architecture style\n\n"
                        "## EVENT-LEVEL\n"
                        "Describe the depicted event or activity:\n"
                        "- What is happening?\n"
                        "- Who appears to be involved?\n"
                        "- What objects or actions are central to the scene?\n\n"
                        "## SCENE-LEVEL\n"
                        "Describe the overall context:\n"
                        "- Indoor or outdoor?\n"
                        "- Time of day and weather conditions (if visible)\n"
                        "- Crowd size and composition\n"
                        "- General atmosphere (tense, celebratory, peaceful, chaotic)"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}",
                    },
                },
            ],
        }
    ]

    resp = llm_provider.chat_completion(messages)
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def get_image_description(
    image_source: Union[str, bytes],
    caption: str = "",
) -> str:
    """
    Multi-granularity visual analysis (EGMMG + EXCLAIM hybrid).

    Local  (Groq): vision LLM structured by entity/event/scene levels.
    Kaggle (vLLM): CLIP zero-shot with dynamic entity probes from caption.
                   CLIP singleton: loaded ONCE, never reloaded.

    Args:
        image_source: URL, local file path (str), or raw bytes.
        caption:      Caption text. Used to extract EGMMG entity probes
                      for CLIP. Empty string = generic probes only.

    Returns:
        Structured English description aligned with DetectiveAgent input format.
    """
    print("[Vision] Loading image...")

    try:
        image_bytes = _load_image_bytes(image_source)
    except ValueError as e:
        print(f"[Vision] WARNING: {e}")
        return "Image could not be loaded. Visual analysis unavailable."

    if Config.IS_KAGGLE:
        print("[Vision] Kaggle -> CLIP multi-granularity (entity probes from caption)...")
        if not caption.strip():
            print("[Vision] WARNING: Caption empty. Entity-level probes will be skipped.")
        return _clip_multigranularity_analysis(image_bytes, caption)

    print("[Vision] Local -> Groq vision LLM (multi-granularity prompt)...")
    return _vision_llm_description(image_bytes)