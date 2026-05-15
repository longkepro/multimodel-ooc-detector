# # import unittest
# # from src.pipeline import run_pipeline

# # class TestPipeline(unittest.TestCase):
# #     def test_full_flow(self):
# #         # Dùng public image + caption fake
# #         result = run_pipeline(
# #             image_url="https://picsum.photos/id/1015/800/600",
# #             caption="This is a fake caption claiming wrong location",
# #             local_image_path=None
# #         )
# #         self.assertIn("verdict", result.lower())
# #         self.assertIn("explanation", result.lower())

# # if __name__ == "__main__":
# #     unittest.main()
# import os
# import base64
# from typing import List, Dict
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field
# from openai import OpenAI
# import instructor
# from serpapi.google_search import GoogleSearch
# import newspaper

# load_dotenv()

# # ====================== CONFIG ======================
# class Config:
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
#     MODEL_NAME = os.getenv("MODEL_NAME", "llama-4-scout-17b-16e-instruct")
#     TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
#     BASE_URL = "https://api.groq.com/openai/v1"

#     # Khi chạy trên Kaggle (sau này) → uncomment 2 dòng dưới để dùng CLIP thật
#     # from sentence_transformers import SentenceTransformer
#     # _encoder = SentenceTransformer('all-MiniLM-L6-v2')

# client = OpenAI(api_key=Config.GROQ_API_KEY, base_url=Config.BASE_URL)
# instructor_client = instructor.from_openai(client)

# # ====================== SCHEMA ======================
# RELATIONS = ["PERFORMS", "LOCATED_IN", "OCCURRED_ON", "TARGETS", "HAS_STATE", "SAME_AS"]

# class SVOTriplet(BaseModel):
#     subject: str = Field(..., description="Main entity")
#     relation: str = Field(..., description=f"One of: {RELATIONS}")
#     object: str = Field(..., description="Value")

# class SVOList(BaseModel):
#     triplets: List[SVOTriplet]

# # ====================== UTILS ======================
# def encode_image(image_path: str) -> str:
#     """Chỉ dùng khi test file local"""
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode()

# def get_image_description(image_url_or_path: str) -> str:
#     """Groq Vision – hỗ trợ cả link public và file local"""
#     if image_url_or_path.startswith(("http://", "https://")):
#         content = [{"type": "image_url", "image_url": {"url": image_url_or_path}}]
#     else:
#         base64_img = encode_image(image_url_or_path)
#         content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]
    
#     resp = client.chat.completions.create(
#         model=Config.MODEL_NAME,
#         messages=[{
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Describe ONLY factual visual details: people, clothing, weather, location, objects, text. No speculation."},
#                 *content
#             ]
#         }],
#         temperature=0.0
#     )
#     return resp.choices[0].message.content

# # ====================== EVIDENCE RETRIEVER (KHÔNG RERANKING - SIÊU NHẸ) ======================
# def retrieve_evidence(image_url: str, caption: str, top_k: int = 5) -> List[Dict]:
#     print("🔍 [1] Google Lens (no reranking - lightweight for laptop)...")
#     # img_desc = get_image_description(image_url)  # nếu muốn dùng vision để query thì uncomment
#     # search_query = f"{img_desc[:200]} {caption}"

#     params = {
#         "engine": "google_lens",
#         "url": image_url,
#         "api_key": Config.SERPAPI_API_KEY,
#         "hl": "en"
#     }
#     results = GoogleSearch(params).get_dict()

#     articles: List[Dict] = []
#     for match in results.get("visual_matches", [])[:top_k * 2]:   # lấy dư để an toàn
#         link = match.get("link")
#         if not link:
#             continue
#         try:
#             article = newspaper.Article(link, language='en')
#             article.download()
#             article.parse()
#             if len(article.text.strip()) > 300:
#                 articles.append({
#                     "url": link,
#                     "title": match.get("title", ""),
#                     "text": article.text[:2500]
#                 })
#         except:
#             continue

#     print(f"✅ Retrieved {len(articles)} evidences from Google Lens.")
#     return articles[:top_k]

# # ====================== SVO EXTRACTOR ======================
# def extract_svo(text: str, source: str = "caption") -> SVOList:
#     print(f"📊 [2] Extracting SVO from {source}...")
#     system = f"""You are a strict knowledge-graph extractor.
# Extract ONLY real triplets that exist in the text.
# Use exactly these relations: {RELATIONS}.
# Never hallucinate."""
#     return instructor_client.chat.completions.create(
#         model=Config.MODEL_NAME,
#         response_model=SVOList,
#         messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
#         temperature=Config.TEMPERATURE,
#     )

# # ====================== AGENTS (giữ nguyên) ======================
# class RetrievalAgent:
#     def __init__(self):
#         self.client = client

#     def _normalize_entity(self, entity: str) -> str:
#         resp = self.client.chat.completions.create(
#             model=Config.MODEL_NAME,
#             messages=[{
#                 "role": "user",
#                 "content": (
#                     "Normalize to canonical English name only. "
#                     "Examples: 'Tổng thống Mỹ'→'Donald Trump', 'Washington'→'Washington DC'.\n"
#                     f"Entity: {entity}\nReturn ONLY the name."
#                 )
#             }],
#             temperature=0.0,
#             max_tokens=30
#         )
#         return resp.choices[0].message.content.strip()

#     def _normalize_svo(self, svo: SVOList) -> List[tuple]:
#         return [
#             (self._normalize_entity(t.subject), t.relation, self._normalize_entity(t.object))
#             for t in svo.triplets
#         ]

#     def run(self, claim_svo: SVOList, evidence_svo: SVOList) -> Dict:
#         print("🔎 [3] Retrieval Agent - Normalize-First...")
#         claim_norm = self._normalize_svo(claim_svo)
#         evidence_norm = self._normalize_svo(evidence_svo)
#         claim_dict = {(sub, rel): obj for sub, rel, obj in claim_norm}

#         inconsistencies = []
#         for sub, rel, obj in evidence_norm:
#             key = (sub, rel)
#             if key in claim_dict and claim_dict[key] != obj:
#                 inconsistencies.append(f"CONFLICT: {sub} {rel} → Claim:'{claim_dict[key]}' vs Evidence:'{obj}'")
#         return {"flagged_inconsistencies": inconsistencies}

# class DetectiveAgent:
#     def run(self, inconsistencies: List[str], image_url_or_path: str, caption: str) -> Dict:
#         print("🕵️  [4] Detective Agent...")
#         img_desc = get_image_description(image_url_or_path)
#         prompt = f"""Image visuals: {img_desc}
# Caption: {caption}
# Inconsistencies: {inconsistencies}
# Analyze which side is correct using visual evidence only."""
#         resp = client.chat.completions.create(
#             model=Config.MODEL_NAME,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0
#         )
#         return {"deep_analysis": resp.choices[0].message.content}

# class AnalystAgent:
#     def run(self, deep_analysis: str) -> str:
#         print("⚖️  [5] Analyst Agent...")
#         prompt = f"""You are the final judge.
# {deep_analysis}

# Return ONLY this JSON:
# {{
#   "verdict": "Real" or "Fake_OOC",
#   "explanation": "detailed natural language explanation in English"
# }}"""
#         resp = client.chat.completions.create(
#             model=Config.MODEL_NAME,
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             temperature=0.0
#         )
#         return resp.choices[0].message.content

# # ====================== PIPELINE ======================
# def run_pipeline(image_url: str, caption: str, local_image_path: str | None = None):
#     print("=" * 70)
#     print("🚀 MULTIMODAL OOC DETECTION PIPELINE (Groq + SerpAPI - ULTRA LIGHT)")
#     print("=" * 70)

#     evidence_list = retrieve_evidence(image_url, caption)
#     if not evidence_list:
#         return "❌ No evidence found."

#     claim_svo = extract_svo(caption, "caption")
#     evidence_svo = extract_svo(evidence_list[0]["text"], "evidence")

#     retrieval = RetrievalAgent().run(claim_svo, evidence_svo)
#     detective = DetectiveAgent().run(retrieval["flagged_inconsistencies"], local_image_path or image_url, caption)
#     result = AnalystAgent().run(detective["deep_analysis"])

#     print("\n🎯 FINAL RESULT:")
#     print(result)
#     return result

# # ====================== CHẠY TEST ======================
# if __name__ == "__main__":
#     # === THAY ĐỔI Ở ĐÂY ===
#     IMAGE_URL = "https://i.ytimg.com/vi/ZtVMoko3mSI/maxresdefault.jpg"   # upload imgbb.com
#     CAPTION = "Donald Trump dancing at France in 2025"
#     LOCAL_PATH = None   # nếu test file local thì điền đường dẫn

#     result = run_pipeline(IMAGE_URL, CAPTION, LOCAL_PATH)

from src.pipeline import run_pipeline

if __name__ == "__main__":
    IMAGE_URL = "https://i.ytimg.com/vi/ZtVMoko3mSI/maxresdefault.jpg"   # thay link thật
    CAPTION = "Donald Trump dancing at France in 2025"
    run_pipeline(IMAGE_URL, CAPTION)