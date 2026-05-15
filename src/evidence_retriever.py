"""
evidence_retriever.py — Stage 1a: Evidence Retrieval & Ranking

Implements the paper's Contextual Items Prediction pipeline (Phase 1):

  Step 1: Google Lens (SerpAPI) → webpage titles + visual entities
  Step 2: Web crawler (newspaper3k) → article text + image captions
  Step 3: CLIP re-ranking using IMAGE embedding as query
          (Paper: "cosine similarity between the image's embedding and the
           textual evidence's embedding, both generated using CLIP")

Key fix vs old code:
  OLD: SentenceTransformer(caption_text) as query  ← WRONG
       Caption is the suspect — using it to rank evidence is circular reasoning.
  NEW: CLIP(image) as query ← CORRECT (paper's approach)
       Image is the ground truth anchor. Find text that best describes the image.
"""

"""
evidence_retriever.py — Stage 1a: Evidence Retrieval & Ranking

FIX: Migrated from legacy `google-search-results` package to new `serpapi` package.

Two separate packages exist — this is the source of the ImportError:
  WRONG (legacy):  pip install google-search-results
                   from serpapi import GoogleSearch          ← breaks on new serpapi
  CORRECT (new):   pip install serpapi
                   import serpapi; client = serpapi.Client() ← official API

Install:
  pip install serpapi newspaper3k transformers torch pillow requests scikit-learn
"""

"""
evidence_retriever.py — Stage 1a: Evidence Retrieval & Ranking

Environment split (Adapter Pattern):
  Local  → Re-rank bằng Groq API (LLM scoring) — không cần torch/CLIP
  Kaggle → Re-rank bằng CLIP image embedding    — cần torch + GPU

Fixes:
  1. torch chỉ import trong Kaggle path → local không cần cài
  2. Visual entities: log raw response để debug field names thực tế
  3. Fallback khi không có article nào vượt qua filter
"""

import numpy as np
from typing import TypedDict

import serpapi
import newspaper

from src.config import Config


# ──────────────────────────────────────────────────────────────
# RETURN TYPE
# ──────────────────────────────────────────────────────────────

class EvidenceItem(TypedDict):
    url:            str
    title:          str
    text:           str
    image_captions: list[str]
    clip_score:     float


# ──────────────────────────────────────────────────────────────
# SERPAPI — Google Lens
# ──────────────────────────────────────────────────────────────

def _google_lens_search(image_url: str) -> dict:
    client = serpapi.Client(api_key=Config.SERPAPI_API_KEY)
    try:
        results = client.search({
            "engine": "google_lens",
            "url":    image_url,
            "hl":     "en",
        })
        return dict(results)
    except serpapi.HTTPError as e:
        if e.status_code == 401:
            raise EnvironmentError("SerpAPI: Invalid API key.") from e
        elif e.status_code == 429:
            raise RuntimeError("SerpAPI: Rate limit exceeded.") from e
        print(f"[Evidence] SerpAPI error {e.status_code}: {e}")
        return {}
    except Exception as e:
        print(f"[Evidence] SerpAPI unexpected error [{type(e).__name__}]: {e}")
        return {}


def _extract_visual_entities(results: dict) -> list[str]:
    """
    Extract visual entities from Google Lens response.

    Google Lens response structure varies — log top-level keys on first call
    so we can see exactly what fields are available in the real response.
    """
    # DEBUG: log available top-level keys (remove after confirming structure)
    top_keys = [k for k in results.keys() if k != "search_metadata"]
    print(f"[Evidence] Google Lens response keys: {top_keys}")

    entities: list[str] = []

    # knowledge_graph (most reliable — direct entity identification)
    kg = results.get("knowledge_graph", {})
    if isinstance(kg, dict) and kg.get("title"):
        entities.append(kg["title"])

    # visual_matches: each match may have its own knowledge_graph or title
    for match in results.get("visual_matches", [])[:10]:
        # Sub-level knowledge_graph
        sub_kg = match.get("knowledge_graph", {})
        if isinstance(sub_kg, dict) and sub_kg.get("title"):
            e = sub_kg["title"].strip()
            if e and e not in entities:
                entities.append(e)

        # Direct title field (short titles are often entity names)
        title = match.get("title", "").strip()
        if title and len(title) < 60 and title not in entities:
            entities.append(title)

    # image_sources (some responses use this instead of visual_matches)
    for src in results.get("image_sources", [])[:5]:
        title = src.get("title", "").strip()
        if title and len(title) < 60 and title not in entities:
            entities.append(title)

    # text_results: titles from web results about the image
    for tr in results.get("text_results", [])[:5]:
        title = tr.get("title", "").strip()
        if title and len(title) < 80 and title not in entities:
            entities.append(title)

    # entity_results (newer Google Lens format)
    for er in results.get("entity_results", [])[:5]:
        name = er.get("name", "").strip() or er.get("title", "").strip()
        if name and name not in entities:
            entities.append(name)

    return entities[:10]


# ──────────────────────────────────────────────────────────────
# WEB CRAWLING
# ──────────────────────────────────────────────────────────────

def _extract_image_captions(article: newspaper.Article) -> list[str]:
    if not article.text:
        return []
    sentences = [s.strip() for s in article.text.split(".") if s.strip()]
    return [s for s in sentences if 10 < len(s) < 150][:5]


def _fetch_article(url: str, title: str) -> dict | None:
    """
    Crawl article content from a URL.

    Reduce min_length to 100 chars to handle short-but-valid news pages.
    Crawl article. min_length reduced to 100 chars to handle
    short-but-valid news pages (original 300 was too strict).
    """
    try:
        # Download article content
        article = newspaper.Article(url, language="en")
        article.download()

        # Parse article content
        article.parse()

        # Extract text content
        text = article.text.strip()
        if len(text) < 100:
            print(f"[Evidence] Too short ({len(text)} chars), skipping: {url}")
            return None

        # Extract image captions
        image_captions = _extract_image_captions(article)

        # Return article content
        return {
            "url":            url,
            "title":          title,
            "text":           text[:2500],
            "image_captions": image_captions,
            "image_captions": _extract_image_captions(article),
            "clip_score":     0.0,
        }
    except Exception as e:
        # Print error message if fetch fails
        print(f"[Evidence] Fetch failed [{type(e).__name__}]: {url} — {e}")
        return None


# ──────────────────────────────────────────────────────────────
# RE-RANKING — LOCAL: Groq API (no torch needed)
# ──────────────────────────────────────────────────────────────

def _rerank_by_groq(
    articles:  list[dict],
    image_url: str,
    top_k:     int,
) -> list[EvidenceItem]:
    """
    Re-rank articles using LLM relevance scoring via Groq API.

    Local environment: no torch, no CLIP, no GPU.
    Ask the LLM to score each article's relevance to the image URL.

    Returns articles sorted by LLM relevance score (0.0–1.0).
    """
    from src.llm_provider import llm_provider

    if len(articles) <= top_k:
        print(f"[Rerank/Groq] {len(articles)} articles ≤ top_k={top_k}, skipping re-rank.")
        for a in articles:
            a["clip_score"] = 0.5   # neutral score
        return articles

    print(f"[Rerank/Groq] Scoring {len(articles)} articles via LLM...")

    # Build candidate list
    candidates = "\n".join(
        f"{i+1}. Title: {a['title']}\n   Snippet: {a['text'][:200]}"
        for i, a in enumerate(articles)
    )

    prompt = f"""You are ranking news articles by relevance to an image.
Image URL: {image_url}

Candidate articles:
{candidates}

Score each article from 0.0 to 1.0 based on how likely it describes
the same event or subject shown in the image.
Return ONLY a JSON array of scores in order, e.g.: [0.9, 0.3, 0.7, 0.1]
No explanation, no markdown."""

    messages = [{"role": "user", "content": prompt}]
    resp = llm_provider.chat_completion(messages)
    raw  = resp.choices[0].message.content.strip()

    # Parse score array
    try:
        import json, re
        match = re.search(r'\[[\d.,\s]+\]', raw)
        scores = json.loads(match.group(0)) if match else []
        if len(scores) != len(articles):
            raise ValueError(f"Expected {len(articles)} scores, got {len(scores)}")
    except Exception as e:
        print(f"[Rerank/Groq] Score parse failed: {e}. Using original order.")
        scores = [1.0 / (i + 1) for i in range(len(articles))]  # fallback: positional decay

    for i, article in enumerate(articles):
        article["clip_score"] = float(scores[i]) if i < len(scores) else 0.0

    # ranked = sorted(articles, key=lambda x: x["clip_score"], reverse=True)[:top_k]
    ranked_all = sorted(articles, key=lambda x: x["clip_score"], reverse=True)
    ranked = ranked_all[:top_k]
    # print(f'[Rerank/Groq] Top scores: {[f"{a["clip_score"]:.2f}" for a in ranked]}')
    scores_str = [f"{a['clip_score']:.2f}" for a in ranked]
    print(f'[Rerank/Groq] Top scores: {scores_str}')
    print("[Rerank/Groq] Top evidence URLs:")
    for idx, ev in enumerate(ranked, start=1):
        print(f"  {idx}. score={ev.get('clip_score', 0.0):.2f} | {ev.get('url', '')}")

    dropped = ranked_all[top_k:]
    if dropped:
        print("[Rerank/Groq] Discarded evidence (outside top_k):")
        for idx, ev in enumerate(dropped, start=1):
            print(f"  {idx}. score={ev.get('clip_score', 0.0):.2f} | {ev.get('url', '')}")

    return ranked


# ──────────────────────────────────────────────────────────────
# RE-RANKING — KAGGLE: CLIP image embedding (needs torch + GPU)
# ──────────────────────────────────────────────────────────────

_clip_model     = None
_clip_processor = None


def _get_clip():
    """Lazy singleton — only called on Kaggle."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        # Import torch only here — local path never reaches this function
        from transformers import CLIPModel, CLIPProcessor
        print("[CLIP] Loading clip-vit-base-patch32 (one-time)...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor


def _rerank_by_clip(
    articles:    list[dict],
    image_bytes: bytes,
    top_k:       int,
) -> list[EvidenceItem]:
    """
    Re-rank using CLIP image embedding (Kaggle/GPU only).
    torch imported inside — local never triggers this path.
    """
    import torch
    from PIL import Image
    from io import BytesIO

    if not articles:
        return []

    model, processor = _get_clip()
    print(f"[CLIP] Re-ranking {len(articles)} articles by image embedding...")

    # Image embedding
    image   = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs  = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    img_emb_np = img_emb[0].numpy()

    # Text embeddings (batch)
    texts  = [f"{a['title']}. {a['text'][:300]}" for a in articles]
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        txt_embs = model.get_text_features(**inputs)
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
    txt_embs_np = txt_embs.numpy()

    scores = txt_embs_np @ img_emb_np
    for i, article in enumerate(articles):
        article["clip_score"] = float(scores[i])

    # ranked = sorted(articles, key=lambda x: x["clip_score"], reverse=True)[:top_k]
    ranked_all = sorted(articles, key=lambda x: x["clip_score"], reverse=True)
    ranked = ranked_all[:top_k]
    # print(f'[Rerank/Groq] Top scores: {[f"{a["clip_score"]:.2f}" for a in ranked]}')
    scores_str = [f"{a['clip_score']:.2f}" for a in ranked]
    print(f'[Rerank/CLIP] Top scores: {scores_str}')
    print("[Rerank/CLIP] Top evidence URLs:")
    for idx, ev in enumerate(ranked, start=1):
        print(f"  {idx}. score={ev.get('clip_score', 0.0):.2f} | {ev.get('url', '')}")

    dropped = ranked_all[top_k:]
    if dropped:
        print("[Rerank/CLIP] Discarded evidence (outside top_k):")
        for idx, ev in enumerate(dropped, start=1):
            print(f"  {idx}. score={ev.get('clip_score', 0.0):.2f} | {ev.get('url', '')}")

    return ranked


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def retrieve_evidence(
    image_url:   str,
    image_bytes: bytes | None = None,
    top_k:       int = Config.MAX_EVIDENCE,
) -> tuple[list[EvidenceItem], list[str]]:
    """
    Stage 1a: Retrieve and rank web evidence for an image.

    Local  → Groq API re-ranking (no torch required)
    Kaggle → CLIP image-embedding re-ranking (torch + GPU)

    Returns:
        (evidence_list, visual_entities)
    """
    print(f"[Evidence] Google Lens: {image_url[:80]}...")

    results = _google_lens_search(image_url)
    if not results:
        return [], []

    visual_entities = _extract_visual_entities(results)
    print(f"[Evidence] Visual entities: {visual_entities}")

    visual_matches = results.get("visual_matches", [])
    if not visual_matches:
        print("[Evidence] No visual_matches in response.")
        return [], visual_entities

    # Crawl articles
    print(f"[Evidence] Crawling up to {top_k * 3} URLs...")
    articles: list[dict] = []
    for match in visual_matches[:top_k * 3]:
        url = match.get("link")
        if not url:
            continue
        article = _fetch_article(url, match.get("title", ""))
        if article:
            articles.append(article)
        if len(articles) >= top_k * 2:
            break

    if not articles:
        print("[Evidence] No usable articles found after crawling.")
        return [], visual_entities

    # Re-rank
    if Config.IS_KAGGLE:
        # Download image for CLIP if not provided
        if image_bytes is None:
            try:
                import requests
                resp = requests.get(image_url, allow_redirects=True, timeout=15,
                                    headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                image_bytes = resp.content
            except Exception as e:
                print(f"[Evidence] Image download failed: {e}. Falling back to Groq re-rank.")
                return _rerank_by_groq(articles, image_url, top_k), visual_entities

        ranked = _rerank_by_clip(articles, image_bytes, top_k)
    else:
        # Local: Groq API — no torch, no CLIP, no GPU
        ranked = _rerank_by_groq(articles, image_url, top_k)

    print(f"[Evidence] Done. {len(ranked)} evidence items returned.")
    return ranked, visual_entities