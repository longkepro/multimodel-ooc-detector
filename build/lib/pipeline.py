"""
pipeline.py — LangGraph Multi-Agent Pipeline (V4: Hybrid Architecture)

Thay đổi hiện tại:
  - Loại bỏ HOÀN TOÀN trích xuất Hints (Date/Location) để chống Anchor Bias.
  - Gộp toàn bộ Title, Text, và Image Captions của bài báo vào làm Raw Text.
  - Pipeline chạy thẳng: Retrieval (Hybrid) -> Detective -> Analyst -> END.
"""

import json
from pathlib import Path
from typing import TypedDict, Any, Optional

from langgraph.graph import StateGraph, END

from src.config import Config
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.detective_agent import DetectiveAgent
from src.agents.analyst_agent   import AnalystAgent


# ──────────────────────────────────────────────────────────────
# STATE 
# ──────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    image_url:              str
    caption:                str
    image_bytes:            Optional[bytes]
    visual_entities:        list[str]
    evidence_list:          list
    evidence_context:       dict  # Giờ đây CHỈ chứa Raw Text
    inconsistencies:        list[str]
    deep_analysis:          str
    final_result:           dict

def _text_rerank(caption: str, evidence_list: list) -> list:
    """Sử dụng LLM để chấm điểm nhanh mức độ liên quan của các bài báo với Caption"""
    if len(evidence_list) <= 2:
        return evidence_list

    from src.llm_provider import llm_provider
    import re, json

    print(f"[Rerank] Đang xếp hạng lại {len(evidence_list)} bài báo bằng LLM...")
    
    # Tạo bảng tóm tắt các bài báo để đưa cho LLM chấm điểm
    snippets = ""
    for i, ev in enumerate(evidence_list):
        title = ev.get('title', 'Unknown')
        text_snippet = ev.get('text', '')[:250] # Lấy 250 ký tự đọc lướt
        snippets += f"[{i}] Title: {title}\nSnippet: {text_snippet}\n\n"

    prompt = f"""You are a Fact-Checking Assistant. 
Evaluate how relevant each article is to fact-checking this CAPTION: "{caption}"

ARTICLES:
{snippets}

Score each article from 0 to 10 based on relevance (10 means it directly addresses or fact-checks the entities/events in the caption).
Return ONLY a JSON array of integers in the exact same order. Example: [10, 0, 5, 8, 2]"""

    try:
        resp = llm_provider.chat_completion([{"role": "user", "content": prompt}])
        raw = resp.choices[0].message.content.strip()
        match = re.search(r'\[[\d,\s]+\]', raw)
        scores = json.loads(match.group(0)) if match else [0] * len(evidence_list)
        
        # Gán điểm và sort lại
        for i, ev in enumerate(evidence_list):
            ev['rerank_score'] = scores[i] if i < len(scores) else 0
            
        ranked = sorted(evidence_list, key=lambda x: x.get('rerank_score', 0), reverse=True)
        return ranked
    except Exception as e:
        print(f"⚠️ [Rerank] Lỗi xếp hạng: {e}. Dùng thứ tự gốc.")
        return evidence_list

# ──────────────────────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────────────────────

def retrieval_node(state: PipelineState) -> dict:
    print("\n─── [Node: Retrieval] ───────────────────────────")

    evidence_list   = state.get("evidence_list", [])
    visual_entities = state.get("visual_entities", [])

    if not evidence_list:
        from src.evidence_retriever import retrieve_evidence
        evidence_list, visual_entities = retrieve_evidence(
            image_url=state["image_url"],
            image_bytes=state.get("image_bytes"),
            top_k=Config.MAX_EVIDENCE,
        )

    raw_caption = state["caption"]

    if evidence_list or visual_entities:
        filtered_ctx = {} 
        
        # 1. GỌI HÀM RERANK TRƯỚC KHI CẮT LẤY TOP 2
        ranked_evidence = _text_rerank(raw_caption, evidence_list)
        
        # 2. GỘP TOÀN BỘ TITLE, TEXT, IMAGE CAPTIONS LÀM CHÂN LÝ TỐI THƯỢNG (Raw Text)
        raw_text_parts = []
        # Lấy tối đa 2 bài báo tốt nhất SAU KHI ĐÃ ĐƯỢC XẾP HẠNG
        for i, ev in enumerate(ranked_evidence[:2]):
            part = f"--- ARTICLE {i+1} ---\n"
            part += f"Title: {ev.get('title', 'Unknown')}\n"
            part += f"Content: {ev.get('text', '')}\n"
            
            # Xử lý Image Captions nếu crawler trả về
            captions = ev.get('image_captions', [])
            if captions:
                if isinstance(captions, list):
                    part += f"Image Captions: {' | '.join(captions)}\n"
                else:
                    part += f"Image Captions: {captions}\n"
                    
            raw_text_parts.append(part)
            
        filtered_ctx["raw_text"] = "\n".join(raw_text_parts)
        
        # ---> HIỂN THỊ LOG DEBUG CHO EVIDENCE TEXT (Giới hạn 800 ký tự) <---
        print("\n[Debug] Trích đoạn bài báo được chọn (Top 2):")
        snippet = filtered_ctx["raw_text"][:800].replace('\n', ' | ')
        print(f"   {snippet}... [truncated]")
        print("─────────────────────────────────────────────────\n")
        
    else:
        filtered_ctx = {"raw_text": "No articles available."}
        print("[Retrieval] No external evidence available.")

    # 3. ĐƯA VÀO CHO RETRIEVAL AGENT (Qwen 2.5)
    inconsistencies = RetrievalAgent().run(raw_caption, filtered_ctx)["flagged_inconsistencies"]

    return {
        "evidence_list":          evidence_list,
        "visual_entities":        visual_entities,
        "evidence_context":       filtered_ctx,
        "inconsistencies":        inconsistencies,
    }

def detective_node(state: PipelineState) -> dict:
    print("\n─── [Node: Detective] ───────────────────────────")
    result = DetectiveAgent().run(
        conflicts=state.get("inconsistencies", []), 
        image_url=state["image_url"],               
        caption=state["caption"]
    )
    return {"deep_analysis": result.get("deep_analysis")}


def analyst_node(state: PipelineState) -> dict:
    print("\n─── [Node: Analyst] ─────────────────────────────")
    result = AnalystAgent().run(state["deep_analysis"])
    return {
        "final_result": result,
    }


# ──────────────────────────────────────────────────────────────
# GRAPH
# ──────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    graph = StateGraph(PipelineState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("detective", detective_node)
    graph.add_node("analyst",   analyst_node)
    
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "detective")
    graph.add_edge("detective", "analyst")
    graph.add_edge("analyst", END)
    
    return graph.compile()

_app = _build_graph()


# ──────────────────────────────────────────────────────────────
# CHECKPOINT & ENTRY POINT
# ──────────────────────────────────────────────────────────────

def _save_checkpoint(sample_id: str, state: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"checkpoint_{sample_id}.json"
    try:
        state_copy = {k: v for k, v in state.items() if k != "image_bytes"}
        with open(path, "w") as f:
            json.dump(state_copy, f, default=str, indent=2)
    except Exception as e:
        print(f"[Checkpoint] Save failed: {e}")


def _load_checkpoint(sample_id: str, out_dir: str) -> Optional[dict]:
    path = Path(out_dir) / f"checkpoint_{sample_id}.json"
    if path.exists():
        with open(path) as f:
            print(f"[Checkpoint] Resuming: {path.name}")
            return json.load(f)
    return None


def run_pipeline(
    image_url:           str,
    caption:             str,
    image_bytes:         bytes | None = None,
    sample_id:           str  = "sample",
    use_checkpoint:      bool = True,
    checkpoint_dir:      str  = "/kaggle/working/checkpoints",
    preloaded_evidence:  list | None = None,
    preloaded_entities:  list | None = None,
) -> dict:
    
    Config.log_env()
    Config.validate()

    if use_checkpoint and Config.IS_KAGGLE:
        cached = _load_checkpoint(sample_id, checkpoint_dir)
        if cached and "final_result" in cached:
            return cached["final_result"]

    mode = "batch (preloaded)" if preloaded_evidence is not None else "demo (SerpAPI)"
    print(f"\n[Pipeline] START | id={sample_id} | mode={mode}")
    print(f"  Image:   {str(image_url)[:80]}")
    print(f"  Caption: {caption[:100]}")

    initial_state: PipelineState = {
        "image_url":              image_url,
        "caption":                caption,
        "image_bytes":            image_bytes,
        "evidence_list":          preloaded_evidence or [],
        "visual_entities":        preloaded_entities or [],
        "evidence_context":       {}, 
        "inconsistencies":        [],
        "deep_analysis":          "",
        "final_result":           {},
    }

    final_state = _app.invoke(initial_state)

    if Config.IS_KAGGLE:
        _save_checkpoint(sample_id, final_state, checkpoint_dir)

    result = final_state["final_result"]
    print(f"\n[Pipeline] DONE: {result.get('verdict')}")
    return result