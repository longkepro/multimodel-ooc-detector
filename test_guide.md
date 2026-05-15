# Testing Guide — Multimodal OOC Detector

## Cấu trúc file cần đảm bảo đúng vị trí

```
multimodal-ooc-detector/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── detective_agent.py
│   │   └── analyst_agent.py        ← file mới
│   ├── config.py                   ← file mới
│   ├── contextual_items_extractor.py ← file mới (thay svo_extractor.py)
│   ├── evidence_retriever.py       ← file mới
│   ├── llm_provider.py             ← file mới
│   ├── pipeline.py                 ← file mới
│   ├── schemas.py                  ← file mới
│   └── utils.py                    ← file mới
├── tests/
│   ├── test_schemas.py             ← viết mới (xem bên dưới)
│   └── test_pipeline_local.py      ← viết mới (xem bên dưới)
├── notebooks/
│   └── kaggle_run.ipynb
├── .env                            ← tạo từ .env.example
└── .env.example
```

---

## PHẦN 1: TEST LOCAL (VSCode / CPU)

### Bước 1 — Cài dependencies

```bash
pip install -r requirements.txt

# Nếu chưa có requirements.txt, cài thủ công:
pip install openai instructor pydantic python-dotenv \
            newspaper3k google-search-results \
            transformers torch pillow requests \
            scikit-learn numpy spacy langgraph

# spaCy model (cần cho entity extraction trong utils.py)
python -m spacy download en_core_web_sm
```

Ghi chú (Windows): `vllm` không phù hợp để cài/chạy trực tiếp trên Windows (thường sẽ lỗi build wheel hoặc thiếu CUDA). Nếu bạn đang test local trên Windows, hãy chạy **API Mode** (Groq) và **không cài `vllm`**. `vllm` chỉ dùng khi chạy trên Kaggle/Linux có GPU.

### Bước 2 — Tạo file .env

```bash
cp .env.example .env
```

Điền vào `.env`:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
SERPAPI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MODEL_NAME=llama-4-scout-17b-16e-instruct
TEMPERATURE=0.0
MAX_EVIDENCE=3
MAX_REFINE=2
REFINE_THRESHOLD=0.75
```

### Bước 3 — Test từng module độc lập

Chạy từng test nhỏ để xác nhận từng layer hoạt động TRƯỚC khi chạy full pipeline.

#### Test 3a: Schemas (không cần API)

```python
# tests/test_schemas.py
from src.schemas import SVOTriplet, SVOList

def test_relation_normalization():
    # Valid canonical
    t = SVOTriplet(subject="Obama", relation="LOCATED_IN", object="Berlin")
    assert t.relation == "LOCATED_IN"

    # Alias normalization
    t = SVOTriplet(subject="Obama", relation="is in", object="Berlin")
    assert t.relation == "LOCATED_IN"

    # Case-insensitive
    t = SVOTriplet(subject="Obama", relation="located_in", object="Berlin")
    assert t.relation == "LOCATED_IN"

    # Unknown → HAS_STATE (not crash)
    t = SVOTriplet(subject="Obama", relation="XYZ_UNKNOWN", object="something")
    assert t.relation == "HAS_STATE"

    print("✅ All schema tests passed")

test_relation_normalization()
```

```bash
python -m pytest tests/test_schemas.py -v
# hoặc
python tests/test_schemas.py
```

#### Test 3b: Config + LLM connection

```python
# Chạy trong Python shell hoặc notebook cell
from src.config import Config
Config.log_env()
Config.validate()   # sẽ raise lỗi nếu thiếu API key

from src.llm_provider import llm_provider
resp = llm_provider.chat_completion([
    {"role": "user", "content": "Reply with exactly: OK"}
])
print(resp.choices[0].message.content)
# Expected: "OK"
```

#### Test 3c: Contextual items extraction (1 API call)

```python
from src.contextual_items_extractor import extract_caption_svo

caption = "Barack Obama giving a speech at the Brandenburg Gate in Berlin, Germany in 2008"
svo, ctx = extract_caption_svo(caption)

print("Context items:", ctx.model_dump())
print("S-V-O triplets:\n", svo.to_display())

# Expected output (approximate):
# Context items: {'people': 'Barack Obama', 'location': 'Brandenburg Gate, Berlin',
#                 'date': '2008', 'event': 'speech', ...}
# S-V-O:
#   Barack Obama --[PERFORMS]--> speech
#   Barack Obama --[LOCATED_IN]--> Brandenburg Gate, Berlin
#   Barack Obama --[OCCURRED_ON]--> 2008
```

#### Test 3d: Evidence retrieval (cần SERPAPI key + internet)

```python
from src.evidence_retriever import retrieve_evidence

# Dùng ảnh test từ NewsCLIPpings hoặc bất kỳ URL ảnh công khai
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/800px-President_Barack_Obama.jpg"

evidence_list, visual_entities = retrieve_evidence(image_url, top_k=2)

print(f"Visual entities: {visual_entities}")
print(f"Evidence count: {len(evidence_list)}")
for ev in evidence_list:
    print(f"  [{ev['clip_score']:.3f}] {ev['title'][:60]}")
```

#### Test 3e: Full pipeline với 1 sample

```python
# tests/test_pipeline_local.py
from src.pipeline import run_pipeline

# Sample thật từ NewsCLIPpings (OOC case)
result = run_pipeline(
    image_url="https://asiasociety.org/sites/default/files/styles/1200w/public/1/120125_obama_state_of_the_union_2.jpg",
    caption="Donald Trump speaking at a press conference in Berlin, Germany, 2019",
    sample_id="test_001",
    use_checkpoint=False,   # không dùng checkpoint khi test
)

print(result)
assert result["verdict"] in ["Real", "Fake_OOC", "Unknown"]
assert 0.0 <= result["confidence"] <= 1.0
print("✅ Pipeline test passed")
```

### Bước 4 — Debug tips

```python
# Nếu CLIP re-ranking chậm trên CPU, test với top_k=1
from src.evidence_retriever import retrieve_evidence
evidence_list, entities = retrieve_evidence(image_url, top_k=1)

# Nếu LLM trả về relation lạ, bật warning trong schemas.py sẽ tự log:
# [Schema] WARNING: Unknown relation 'IS_AT'. Defaulting to HAS_STATE.

# Kiểm tra SVO output trực tiếp
from src.contextual_items_extractor import extract_contextual_svo
svo, ctx = extract_contextual_svo(evidence_list, entities)
print(svo.to_display())
```

---

## PHẦN 2: TEST TRÊN KAGGLE

### Bước 1 — Chuẩn bị Kaggle Secrets

Vào **Notebook Settings → Add-ons → Secrets** và thêm:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | (để trống — không dùng trên Kaggle) |
| `SERPAPI_API_KEY` | key thật của bạn |
| `KAGGLE_MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` |
| `TENSOR_PARALLEL_SIZE` | `1` (T4 x1) hoặc `2` (T4 x2) |

### Bước 2 — Thêm datasets vào notebook

Trong notebook settings → **Add Data**:
1. `g-luo/news-clippings` (NewsCLIPpings dataset)
2. Model dataset 4-bit quantized (nếu dùng local model)
   - Hoặc dùng Groq API key nếu không muốn tải model

### Bước 3 — Notebook setup cell

```python
# Cell 1: Install dependencies
!pip install -q openai instructor pydantic python-dotenv \
               newspaper3k google-search-results \
               transformers torch pillow requests \
               scikit-learn numpy spacy langgraph vllm

# Lưu ý: Cell này chỉ dành cho Kaggle/Linux. Nếu bạn chạy trên Windows local,
# việc cài `vllm` có thể fail (pip phải build wheel) và gây lỗi kiểu:
# "Failed building wheel for vllm" hoặc lỗi đường dẫn quá dài.

!python -m spacy download en_core_web_sm -q

# Cell 2: Load Kaggle secrets vào environment
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["SERPAPI_API_KEY"]    = secrets.get_secret("SERPAPI_API_KEY")
os.environ["KAGGLE_MODEL_NAME"]  = secrets.get_secret("KAGGLE_MODEL_NAME")
os.environ["TENSOR_PARALLEL_SIZE"] = secrets.get_secret("TENSOR_PARALLEL_SIZE")

# Cell 3: Clone/copy code vào working directory
# Option A: từ GitHub
!git clone https://github.com/your-repo/multimodal-ooc-detector.git
%cd multimodal-ooc-detector

# Option B: upload code trực tiếp lên Kaggle Dataset và mount vào
import sys
sys.path.append("/kaggle/input/your-code-dataset")
```

### Bước 4 — Verify Kaggle environment

```python
# Cell 4: Check config
from src.config import Config
Config.log_env()
# Expected: 🚀 Kaggle detected → vLLM 4-bit | model=...

Config.validate()
print("✅ Config OK")
```

### Bước 5 — Batch processing với checkpoint

```python
# Cell 5: Load dataset
import json
import pandas as pd

# NewsCLIPpings test split
with open("/kaggle/input/news-clippings/news_clippings/data/merged_balanced/test.json") as f:
    test_data = json.load(f)

# Visual News metadata (for image URLs)
with open("/kaggle/input/visual-news/origin/data.json") as f:
    visual_news = json.load(f)

df_test = pd.DataFrame(test_data["annotations"][:100])  # bắt đầu với 100 mẫu
print(f"Test samples: {len(df_test)}")
```

```python
# Cell 6: Run batch với checkpoint (resume-safe)
from src.pipeline import run_pipeline
from pathlib import Path
import pandas as pd

results = []
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

for i, row in df_test.iterrows():
    sample_id = str(row["id"])
    checkpoint_path = Path(CHECKPOINT_DIR) / f"checkpoint_{sample_id}.json"

    # Skip nếu đã có kết quả
    if checkpoint_path.exists():
        import json
        result = json.load(open(checkpoint_path))
        if "final_result" in result:
            results.append({**result["final_result"], "id": sample_id, "true_label": row["falsified"]})
            continue

    # Lấy image URL từ VisualNews
    image_meta = visual_news.get(str(row["image_id"]), {})
    image_url  = image_meta.get("image_url", "")

    if not image_url:
        print(f"[{i}] No image URL for {sample_id}, skipping")
        continue

    try:
        result = run_pipeline(
            image_url=image_url,
            caption=row["caption"],
            sample_id=sample_id,
            use_checkpoint=True,
        )
        results.append({**result, "id": sample_id, "true_label": row["falsified"]})
    except Exception as e:
        print(f"[{i}] ERROR on {sample_id}: {e}")
        continue

    if i % 10 == 0:
        print(f"Progress: {i}/{len(df_test)}")
```

```python
# Cell 7: Evaluate
from sklearn.metrics import classification_report, accuracy_score

df_results = pd.DataFrame(results)

# Map: falsified=True → Fake_OOC, falsified=False → Real
df_results["true_verdict"] = df_results["true_label"].map({True: "Fake_OOC", False: "Real"})
df_results_clean = df_results[df_results["verdict"].isin(["Real", "Fake_OOC"])]

print(f"Valid predictions: {len(df_results_clean)}/{len(df_results)}")
print(f"\nAccuracy: {accuracy_score(df_results_clean['true_verdict'], df_results_clean['verdict']):.3f}")
print("\nClassification Report:")
print(classification_report(df_results_clean["true_verdict"], df_results_clean["verdict"]))

# Save results
df_results.to_csv("/kaggle/working/results.csv", index=False)
```

---

## Checklist trước khi submit nghiên cứu

- [ ] `test_schemas.py` pass toàn bộ
- [ ] Pipeline chạy được trên 5 mẫu local không có lỗi
- [ ] CLIP singleton không reload (kiểm tra log: "[CLIP] Loading..." chỉ xuất hiện 1 lần)
- [ ] Không có `"Unknown"` verdict trong >10% kết quả (nếu có nhiều → LLM parse đang fail)
- [ ] `confidence` distribution hợp lý (không phải toàn 0.6 → LLM đang dùng fallback)
- [ ] Checkpoint files được tạo trong `/kaggle/working/`
- [ ] Kết quả được lưu ra `results.csv` trước khi session timeout