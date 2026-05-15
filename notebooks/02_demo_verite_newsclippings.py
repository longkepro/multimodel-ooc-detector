import pandas as pd
from src.pipeline import run_pipeline

# 5 sample đại diện (VERITE + NewsCLIPpings style)
# Bạn có thể thay image_url bằng link thật từ VERITE GitHub[](https://github.com/stevejpapad/image-text-verification)
samples = [
    {
        "id": 1,
        "dataset": "VERITE",
        "image_url": "https://picsum.photos/id/133/1024/768",   # railway bridge example
        "caption": "Image shows electric green scooters that have reached the end of their battery life. Due to the batteries being so expensive to replace, electric scooters are abandoned because disposing of them any other way is dangerous and expensive..",
        "ground_truth": "Fake_OOC"
    },
    {
        "id": 2,
        "dataset": "NewsCLIPpings",
        "image_url": "https://mediaproxy.snopes.com/width/1200/https://media.snopes.com/2022/11/3d-rendering-andre-the-giant.png",
        "caption": "Donald Trump dancing at France in 2025",
        "ground_truth": "Fake_OOC"
    },
    {
        "id": 3,
        "dataset": "VERITE",
        "image_url": "https://mediaproxy.snopes.com/width/1200/https://media.snopes.com/2022/11/model.jpg",
        "caption": "Image shows unnamed runway model showcasing designs from Chinese designer and performance artist Sheguang Hu's collection during the Mercedes-Benz China Fashion Week.",
        "ground_truth": "Real"
    },
    {
        "id": 4,
        "dataset": "NewsCLIPpings",
        "image_url": "https://picsum.photos/id/201/1024/768",
        "caption": "Lionel Messi scoring the winning goal in the 2022 World Cup final in Qatar.",
        "ground_truth": "Fake_OOC"
    },
    {
        "id": 5,
        "dataset": "VERITE",
        "image_url": "https://mediaproxy.snopes.com/width/1200/https://media.snopes.com/2022/11/3d-rendering-andre-the-giant.png",
        "caption": "A sketchfab 3D rendering of the actor Andre the Giant.",
        "ground_truth": "Real"
    }
]

results = []
for s in samples:
    print(f"\n{'='*60}\nTesting Sample {s['id']} ({s['dataset']})")
    final = run_pipeline(s["image_url"], s["caption"])
    
    results.append({
        "ID": s["id"],
        "Dataset": s["dataset"],
        "Caption (short)": s["caption"][:60] + "...",
        "Predicted": final["verdict"],
        "Confidence": f"{final['confidence']:.2f}",
        "Ground Truth": s["ground_truth"],
        "Correct": "✓" if final["verdict"] == s["ground_truth"] else "✗",
        "Explanation (short)": final["explanation"][:80] + "..."
    })

# Bảng kết quả đẹp
df = pd.DataFrame(results)
print("\n" + "="*80)
print(" KẾT QUẢ DEMO 5 SAMPLE VERITE + NEWSCLIPPINGS")
print("="*80)
print(df.to_string(index=False))

# Lưu bảng ra Excel (tùy chọn)
df.to_excel("demo_results.xlsx", index=False)
print("\n Đã lưu kết quả vào demo_results.xlsx")