import os
from src.dataset_evidence_loader import VERITELoader, pre_crawl_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))

# TRUYỀN THẲNG FILE GỐC (VERITE.csv) THAY VÌ FILE ĐÃ GỘP
loader = VERITELoader(
    verite_csv=os.path.join(current_dir, "VERITE.csv"), 
    articles_csv=os.path.join(current_dir, "VERITE_articles.csv"),
    image_dir=os.path.join(current_dir, "images/"),
    cache_dir=os.path.join(current_dir, ".crawl_cache/verite"),
)

print(f"🚀 Đang bắt đầu Pre-crawl...")
pre_crawl_dataset(loader, max_samples=1000, delay=1.5)