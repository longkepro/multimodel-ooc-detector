import os
import json
from pathlib import Path

# Đường dẫn đến thư mục cache của bạn
cache_dir = Path("D:/AI Project/multimodel-ooc-detector/data/VERITE/.crawl_cache/vision")

# 1. Đếm số lượng file thực tế
json_files = list(cache_dir.glob("*.json"))
print(f"Tổng số file .json đang có vật lý trên ổ cứng: {len(json_files)}")

# 2. Đếm số lượng file "sạch" (có bằng chứng thật sự)
valid_files = 0
empty_files = 0

for file in json_files:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("evidence_count", 0) > 0:
            valid_files += 1
        else:
            empty_files += 1
    except Exception:
        pass

print(f"Số file CHỨA BẰNG CHỨNG XỊN (evidence > 0): {valid_files}")
print(f"Số file RỖNG (evidence = 0): {empty_files}")