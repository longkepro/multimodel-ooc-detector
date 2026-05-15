"""
retrieval_agent.py — Context Difference Reporter (Chain-of-Thought JSON)

Vai trò: So sánh Raw Caption tự do và Evidence (Toàn bộ Raw Text).
Sử dụng kỹ thuật Chain-of-Thought (CoT) để ép LLM phân tích tách bạch, chống ảo giác (Hallucination) 
do đọc nhầm các trích dẫn tin giả trong bài báo Fact-check và chống bắt bẻ ngữ pháp/từ vựng.
"""

import json
import re
from typing import Dict, List
from src.agents.base_agent import BaseAgent

class RetrievalAgent(BaseAgent):
    def _detect_differences(self, raw_caption: str, evidence_context: dict) -> List[str]:
        
        # Chỉ lấy Raw Text, vứt bỏ toàn bộ logic Hints gây nhiễu
        raw_text = evidence_context.get("raw_text", "No raw text available.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Fact-Checking Analyst. "
                    "Compare the 'RAW CAPTION' against the 'FULL RAW ARTICLES'.\n\n"
                    "CRITICAL RULES - AVOID HALLUCINATIONS & NITPICKING:\n"
                    "1. THE ARTICLES ARE THE IMAGE's GROUND TRUTH (OUT-OF-CONTEXT RULE): The provided articles explain what the image ACTUALLY shows. If the RAW CAPTION claims the image is about Event A (e.g., quarantine enforcement), but the articles prove the image is actually about Event B (e.g., nationwide protests), this is a massive MUTUALLY EXCLUSIVE contradiction. Flag it immediately!\n"
                    "2. BEWARE OF QUOTED FAKE NEWS: Extract ONLY the TRUE FACTS from the article to compare with the RAW CAPTION. Ignore viral fake claims mentioned in the article.\n"
                    "3. FLAG FUNDAMENTAL CONTRADICTIONS ONLY: Flag ONLY if the core narrative, date, or exact event location CANNOT BOTH BE TRUE.\n"
                    "4. DO NOT NITPICK SEMANTICS: 'Taken away' and 'forcibly dragged' are COMPATIBLE. Do not flag differences that are merely variations in vocabulary or phrasing.\n"
                    "5. CONTEXTUALIZE PREPOSITIONS: Distinguish between origins/destinations ('from/to') and the actual physical location ('in').\n"
                    "6. IGNORE MISSING INFO: Lack of explicit proof in one source is not a contradiction. If a detail is simply unmentioned, do not flag it.\n\n"
                    "OUTPUT FORMAT:\n"
                    "You MUST use Chain-of-Thought reasoning by returning a JSON object with EXACTLY these keys:\n"
                    "{\n"
                    "  \"step1_caption_claim\": \"Summarize the holistic, main point of the RAW CAPTION.\",\n"
                    "  \"step2_article_truth\": \"Summarize the TRUE FACTS from the articles.\",\n"
                    "  \"step3_compatibility_analysis\": \"Are they describing the same core event? If the caption describes an entirely different event from what the articles say the image shows, flag it as a contradiction. Otherwise, check for genuine conflicts vs semantics.\",\n"
                    "  \"differences\": [\"[MUTUALLY EXCLUSIVE] The caption claims X, but the true fact is strictly Y.\"]\n"
                    "}\n\n"
                    "If Step 3 concludes they are compatible, the 'differences' array MUST be exactly []."
                )
            },
            {
                "role": "user",
                "content": f"--- RAW CAPTION ---\n{raw_caption}\n\n--- FULL RAW ARTICLES ---\n{raw_text}"
            }
        ]

        resp = self.llm.chat_completion(messages)
        raw = resp.choices[0].message.content.strip()

        # Dọn dẹp markdown rác
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            # Lớp bảo vệ 1: Parse JSON Object mới (Chain of Thought)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                
                # ---> IN RA QUÁ TRÌNH TƯ DUY ĐỂ DEBUG <---
                print(f"   [CoT] Bước 1 (Hiểu Caption): {data.get('step1_caption_claim', '')}")
                print(f"   [CoT] Bước 2 (Hiểu Sự thật): {data.get('step2_article_truth', '')}")
                print(f"   [CoT] Bước 3 (Biện luận): {data.get('step3_compatibility_analysis', '')}")
                
                differences = data.get("differences", [])
                if isinstance(differences, list):
                    return differences
        except Exception as e:
            print(f"⚠️ [Retrieval] JSON parse error: {e}. Kích hoạt Fallback Parser...")
            
        # Lớp bảo vệ 2: Cứu hộ (Fallback) khi JSON sập
        fallback_diffs = []
        for line in raw.split('\n'):
            if "[MUTUALLY EXCLUSIVE]" in line.upper() or "[DIFFERENCE]" in line.upper():
                clean_line = re.sub(r'^.*?(\[MUTUALLY EXCLUSIVE\]|\[DIFFERENCE\])', r'\1', line, flags=re.IGNORECASE)
                clean_line = clean_line.strip('", \']')
                # Ép bọc tag [MUTUALLY EXCLUSIVE] để Gemma 4 không dám cãi
                if not clean_line.startswith("[MUTUALLY EXCLUSIVE]"):
                    clean_line = "[MUTUALLY EXCLUSIVE] " + clean_line.replace("[DIFFERENCE]", "").strip()
                fallback_diffs.append(clean_line)
        
        return fallback_diffs

    def run(self, raw_caption: str, evidence_context: dict) -> Dict:
        print("🔎 [Retrieval] Cross-examining Raw Caption vs Raw Articles...")

        differences = self._detect_differences(raw_caption, evidence_context)

        # Lọc trùng lặp và giới hạn số lượng (Chống tràn Context cho Gemma 4)
        unique_diffs = []
        seen = set()
        for d in differences:
            # Nếu LLM không chịu nhả tag, ta ép thêm vào để khống chế Gemma 4
            if not d.startswith("[MUTUALLY EXCLUSIVE]"):
                d = f"[MUTUALLY EXCLUSIVE] {d}"
                
            key = d[:40].lower() 
            if key not in seen:
                seen.add(key)
                unique_diffs.append(d)
        
        final_diffs = unique_diffs[:5]

        if final_diffs:
            print(f"⚠️  [Retrieval] Found {len(final_diffs)} critical contradiction(s):")
            for d in final_diffs:
                print(f"   {d}")
        else:
            print("✅ [Retrieval] No mutually exclusive contradictions found.")

        return {"flagged_inconsistencies": final_diffs}