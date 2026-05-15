# pre_crawl_evidence.py
import os, json, time, csv, argparse, requests, hashlib
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

import trafilatura
from trafilatura.settings import use_config

import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import requests

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR       = Path(".crawl_cache/vision")
USAGE_FILE      = Path(".vision_usage.json")
MAX_QUOTA       = 2000          # dừng trước 1000/tháng/project
MAX_PAGES       = 5            
CRAWL_TIMEOUT   = 5
MAX_CHARS_TEXT  = 1500   # Sapo + 2 đoạn đầu = đủ Who/What/When/Where
BATCH_SLEEP     = 0.4           # giây giữa các Vision API call
VISION_SCOPE    = ["https://www.googleapis.com/auth/cloud-vision"]
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Trafilatura config: tắt log spam
_traf_cfg = use_config()
_traf_cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "8")

# Đọc danh sách file JSON từ env
SA_FILES = [v for v in [
    os.getenv("VISION_SA_1"),
    os.getenv("VISION_SA_2"),
    os.getenv("VISION_SA_3"),
] if v and Path(v).exists()]

assert SA_FILES, (
    "Không tìm thấy Service Account JSON nào! "
    "Kiểm tra VISION_SA_1 / VISION_SA_2 trong .env"
)

# ── Service Account Key Manager ───────────────────────────────────────────────
class VisionKeyManager:
    """
    Xoay giữa nhiều Service Account JSON files.
    Mỗi file tương ứng 1 Google Cloud project = 1000 request/tháng miễn phí.
    """
    def __init__(self):
        self.sa_files  = SA_FILES
        self.idx       = 0
        self.usage     = self._load_usage()
        self._credentials = {}   # cache credentials đã load

    # ── Persistence ──────────────────────────────────────────────────────────
    def _load_usage(self) -> dict:
        if USAGE_FILE.exists():
            return json.loads(USAGE_FILE.read_text())
        return {f: 0 for f in self.sa_files}

    def _save_usage(self):
        USAGE_FILE.write_text(json.dumps(self.usage, indent=2))

    # ── Credentials ──────────────────────────────────────────────────────────
    def _get_credentials(self, sa_file: str):
        """Load + cache credentials, tự refresh token khi hết hạn."""
        if sa_file not in self._credentials:
            creds = service_account.Credentials.from_service_account_file(
                sa_file, scopes=VISION_SCOPE
            )
            self._credentials[sa_file] = creds
        creds = self._credentials[sa_file]
        # Refresh nếu token sắp hết hạn
        if not creds.valid:
            creds.refresh(google.auth.transport.requests.Request())
        return creds

    @property
    def current_sa(self) -> str:
        return self.sa_files[self.idx]

    def get_token(self) -> str:
        """Trả Bearer token của SA hiện tại."""
        creds = self._get_credentials(self.current_sa)
        return creds.token

    def consume(self):
        """Ghi nhận 1 request đã dùng, tự xoay SA nếu gần hết quota."""
        key = self.current_sa
        self.usage[key] = self.usage.get(key, 0) + 1
        self._save_usage()
        if self.usage[key] >= MAX_QUOTA:
            self._rotate()

    def _rotate(self):
        self.idx += 1
        if self.idx >= len(self.sa_files):
            raise RuntimeError(
                f"Đã dùng hết tất cả {len(self.sa_files)} Service Account!\n"
                f"Usage: {json.dumps(self.usage, indent=2)}"
            )
        new_sa = Path(self.sa_files[self.idx]).stem
        print(f"\n[KEY ROTATE] Chuyển sang Service Account: {new_sa}\n")

    def total_used(self) -> int:
        return sum(self.usage.values())

    def status(self) -> str:
        cur = Path(self.current_sa).stem
        used = self.usage.get(self.current_sa, 0)
        return f"SA={cur} ({used}/{MAX_QUOTA}) | total={self.total_used()}"

# ── Google Vision API call ────────────────────────────────────────────────────
def call_vision_api(image_url: str, key_mgr: VisionKeyManager) -> dict | None:
    """
    Gọi WEB_DETECTION với Bearer token từ Service Account.
    Trả về webDetection dict hoặc None nếu lỗi.
    """
    token = key_mgr.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    body = {
        "requests": [{
            "image":    {"source": {"imageUri": image_url}},
            "features": [{"type": "WEB_DETECTION", "maxResults": 30}]
        }]
    }
    try:
        resp = requests.post(
            VISION_ENDPOINT, json=body,
            headers=headers, timeout=15
        )
        resp.raise_for_status()
        result = resp.json()

        # Kiểm tra lỗi cấp application
        response_obj = result.get("responses", [{}])[0]
        if "error" in response_obj:
            print(f"  [VISION ERR] {response_obj['error']}")
            return None

        key_mgr.consume()
        return response_obj.get("webDetection", {})

    except requests.HTTPError as e:
        code = e.response.status_code if e.response else 0
        if code == 429:
            print(f"  [RATE LIMIT] ngủ 5s...")
            time.sleep(5)
        elif code == 401:
            # Token hết hạn → force refresh
            print(f"  [AUTH] Token expired, refreshing...")
            key_mgr._credentials.pop(key_mgr.current_sa, None)
        else:
            print(f"  [HTTP {code}] {e}")
        return None
    except Exception as e:
        print(f"  [ERR] Vision: {e}")
        return None

# ── Trafilatura crawler ───────────────────────────────────────────────────────
def crawl_page_trafilatura(url: str) -> dict:
    """
    Trafilatura extract structured content.
    Trả dict thay vì string — giữ lại title, description, image_captions
    vì đây là signal quan trọng cho OOC detection.
    """
    empty = {"title": "", "description": "", "text": "", "image_captions": []}
    try:
        # 2. DÙNG REQUESTS ĐỂ LÀM LÍNH TIÊN PHONG (THAY CHO FETCH_URL)
        headers = {
            # Ngụy trang thành trình duyệt Chrome thật để không bị web chặn
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Timeout 10 giây: Nếu server lề mề hoặc bắt đợi, ép buộc ngắt kết nối!
        response = requests.get(url, headers=headers, timeout=10)
        
        # Nếu mã HTTP không phải 200 (OK) -> Bỏ qua luôn
        if response.status_code != 200:
            print(f"  [Skip] Trang web chặn (HTTP {response.status_code}): {url}")
            return empty
            
        downloaded = response.text # Lấy mã nguồn HTML thô

        if not downloaded:
            return empty

        # 3. ĐƯA HTML THÔ CHO TRAFILATURA BÓC TÁCH (PHẦN NÀY GIỮ NGUYÊN)
        metadata = trafilatura.extract_metadata(downloaded)
        desc = metadata.description if (metadata and metadata.description) else ""
        
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        ) or ""

        # Cứu cánh cho các trang JS-Render
        if not text and desc:
            text = desc
            
        if not text.strip():
            return empty

        image_captions = _extract_image_captions(downloaded)

        return {
            "title":          (metadata.title if metadata else ""),
            "description":    desc,
            "text":           text, 
            "image_captions": image_captions[:5],
        }
        
    # Bắt lỗi Timeout (quá 10s) hoặc lỗi mạng
    except requests.exceptions.RequestException as e:
        print(f"  [Skip] Lỗi kết nối/Timeout ({url})")
        return empty
    except Exception as e:
        print(f"  [Skip] Lỗi phân tích ({url}): {e}")
        return empty

def _extract_image_captions(html: str) -> list[str]:
    """
    Lấy <figcaption> và <img alt> — nơi chứa caption ảnh báo chí.
    Đây là ground truth tự nhiên: báo gốc viết caption gì cho ảnh này.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        captions = []

        # <figcaption> — chuẩn nhất trong bài báo
        for tag in soup.find_all("figcaption"):
            t = tag.get_text(strip=True)
            if len(t) > 15:           # lọc caption quá ngắn ("AFP", "Getty")
                captions.append(t)

        # <img alt> — fallback
        for img in soup.find_all("img", alt=True):
            alt = img["alt"].strip()
            if len(alt) > 20:
                captions.append(alt)

        # Dedup giữ thứ tự
        seen, result = set(), []
        for c in captions:
            if c not in seen:
                seen.add(c); result.append(c)
        return result
    except Exception:
        return []

# ── Perceptual hash để verify ảnh trên trang ──────────────────────────────────
def url_hash(url: str) -> str:
    """Simple hash để track URL đã crawl."""
    return hashlib.md5(url.encode()).hexdigest()[:8]

# ── Build cache entry cho 1 sample ────────────────────────────────────────────
def build_cache_entry(
    sample_id: str,
    image_url: str,
    key_mgr: VisionKeyManager
) -> dict | None:
    """
    1. Gọi Vision API → lấy webEntities + pagesWithMatchingImages
    2. Lọc bỏ sạch sẽ các domain Mạng Xã Hội (Blacklist) từ tuyến đầu.
    3. Crawl top-MAX_PAGES link "sạch" nhất bằng Trafilatura.
    4. Ép lỗi (return None) nếu không thu được chữ nào để ép hệ thống cào lại/lưu fail.
    """
    # ── Step 1: Vision API ────────────────────────────────────────────────────
    wd = call_vision_api(image_url, key_mgr)
    if wd is None:
        return None

    visual_entities = [
        {"description": e.get("description", ""), "score": e.get("score", 0)}
        for e in wd.get("webEntities", [])
        if e.get("score", 0) > 0.3
    ]
    best_guess_labels = [
        g["label"] for g in wd.get("bestGuessLabels", [])
    ]
    
    similar_image_urls = [
        img.get("url", "")
        for img in (
            wd.get("fullMatchingImages", []) +
            wd.get("partialMatchingImages", [])
        )
    ][:10]

    # ── Step 2: Over-fetching & Blacklist Filter (TUYẾN ĐẦU) ──────────────────
    BLACKLIST = [
        "youtube.com", "youtu.be", "instagram.com", "pinterest.com", 
        "buymeacoffee.com", "reddit.com", "x.com", "twitter.com",
        "facebook.com", "fb.com", "fb.watch", "tiktok.com"
    ]
    
    all_pages = wd.get("pagesWithMatchingImages", [])
    clean_pages = []
    
    # Lọc lấy tất cả những trang KHÔNG dính Blacklist
    for page in all_pages:
        page_url = page.get("url", "")
        if not page_url:
            continue
            
        if not any(domain in page_url.lower() for domain in BLACKLIST):
            clean_pages.append(page)

    # Lấy ra số lượng trang sạch cần thiết (VD: 5 trang)
    pages_to_crawl = clean_pages[:MAX_PAGES]

    # Chốt chặn 1: Vision API toàn trả về link rác
    if not pages_to_crawl:
        print("  [Warn] Vision API chỉ trả về MXH rác, không có link báo chí hợp lệ!")
        return None

    # ── Step 3: Crawl các trang sạch ──────────────────────────────────────────
    evidence = []

    for page in pages_to_crawl:
        page_url   = page.get("url", "")
        page_title = page.get("pageTitle", "")

        crawled = crawl_page_trafilatura(page_url)

        # Bỏ qua trang nếu trắng tinh không lấy được gì (đã gồm cả xử lý description bù text ở Trafilatura)
        if not any([crawled.get("text"), crawled.get("title"), crawled.get("description")]):
            continue

        evidence.append({
            "url":            page_url,
            "page_title":     page_title,       # từ Vision API
            "title":          crawled.get("title", ""),  # từ Trafilatura
            "description":    crawled.get("description", ""),
            "text":           crawled.get("text", ""),
            "image_captions": crawled.get("image_captions", []),
        })

    # ── Step 4: Chốt chặn 2 - Ép lỗi nếu cào về "tay trắng" ──────────────────
    if len(evidence) == 0:
        print("  [Warn] Đã thử cào nhưng không lấy được Text/Bằng chứng hợp lệ!")
        return None

    # ── Step 5: Đóng gói thành phẩm ───────────────────────────────────────────
    return {
        "sample_id":          sample_id,
        "image_url":          image_url,
        "visual_entities":    visual_entities,
        "best_guess_labels":  best_guess_labels,
        "similar_image_urls": similar_image_urls,
        "evidence":           evidence,
        "evidence_count":     len(evidence),
        "crawled_at":         datetime.now(timezone.utc).isoformat(),
    }

# ── Load VERITE samples ───────────────────────────────────────────────────────
def load_verite_samples(csv_path: str, articles_csv: str, max_samples: int) -> list[dict]:
    # 1. Map cả true_url và false_url (Giống như mình đã bàn)
    url_map_true: dict[str, str] = {}
    url_map_false: dict[str, str] = {}
    
    with open(articles_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_str = str(row["id"])
            url_map_true[id_str] = row.get("true_url", "")
            url_map_false[id_str] = row.get("false_url", "")

    samples = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = row.get("image_path", "")
            
            # Lấy toàn bộ tên file (VD: "true_0" hoặc "false_0")
            img_stem = Path(img_path).stem 
            
            # Lấy số id gốc (VD: "0") để tra cứu URL trong file articles_csv
            base_id = img_stem.split("_")[-1] 
            
            # 2. Logic rẽ nhánh URL an toàn
            if "true_" in img_stem:
                image_url = url_map_true.get(base_id, "")
            else:
                image_url = url_map_false.get(base_id, "")
                
            if not image_url:
                continue
                
            samples.append({
                "id":        img_stem, # BẮT BUỘC dùng img_stem ("true_0") làm ID để lưu Cache không bị đè!
                "caption":   row.get("caption", ""),
                "image_url": image_url,
                "label":     row.get("label", ""),
            })
            
            if len(samples) >= max_samples:
                break
                
    return samples
# ── Main crawl loop ───────────────────────────────────────────────────────────
def run_crawl(dataset: str, max_samples: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key_mgr = VisionKeyManager()

    if dataset == "verite":
        samples = load_verite_samples(
            csv_path     = "VERITE.csv",
            articles_csv = "VERITE_articles.csv",
            max_samples  = max_samples,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset}' chưa hỗ trợ")

    total   = len(samples)
    success = 0
    skipped = 0
    failed  = []

    print(f"\n{'='*60}")
    print(f"Dataset  : {dataset.upper()} — {total} mẫu")
    print(f"SA files : {[Path(f).stem for f in SA_FILES]}")
    print(f"Max pages: {MAX_PAGES} pages/ảnh (theo paper)")
    print(f"Cache    : {CACHE_DIR.resolve()}")
    print(f"{'='*60}\n")

    for i, s in enumerate(samples, 1):
        cache_path = CACHE_DIR / f"{s['id']}.json"

        # ── Resume: skip nếu đã cache ─────────────────────────────────────
        if cache_path.exists():
            skipped += 1
            if i % 100 == 0:
                print(f"[{i:4}/{total}] ... (đã skip {skipped} cached) "
                      f"| {key_mgr.status()}")
            continue

        # ── Crawl ─────────────────────────────────────────────────────────
        print(f"[{i:4}/{total}] id={s['id']:>6} | {key_mgr.status()} | ",
              end="", flush=True)

        entry = build_cache_entry(s["id"], s["image_url"], key_mgr)

        if entry is None:
            print("FAIL")
            failed.append({"id": s["id"], "url": s["image_url"]})
            time.sleep(1)
            continue

        cache_path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        success += 1
        ev_count = entry["evidence_count"]
        ent_count = len(entry["visual_entities"])
        print(f"OK | {ev_count:2} pages, {ent_count:2} entities")

        time.sleep(BATCH_SLEEP)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Thành công : {success}")
    print(f"Đã cached  : {skipped}")
    print(f"Thất bại   : {len(failed)}")
    print(f"API đã dùng: {key_mgr.total_used()}")
    print(f"{'='*60}")

    if failed:
        fail_path = Path(".crawl_failed.json")
        fail_path.write_text(json.dumps(failed, indent=2))
        print(f"→ Lưu danh sách thất bại: {fail_path}")
        print(f"→ Retry: python pre_crawl_evidence.py --dataset {dataset} "
              f"(tự động skip đã cache)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="verite")
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()
    run_crawl(args.dataset, args.max_samples)