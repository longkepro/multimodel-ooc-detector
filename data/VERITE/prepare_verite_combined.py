import pandas as pd
import re
import os

# 1. Xác định đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
df_main     = pd.read_csv(os.path.join(current_dir, "VERITE.csv"))
df_articles = pd.read_csv(os.path.join(current_dir, "VERITE_articles.csv"))

# 2. Tạo cột ID để gộp
df_main["id"] = df_main["image_path"].apply(
    lambda p: int(m.group(1)) if (m := re.search(r'_(\d+)\.', str(p))) else None
)

# 3. Gộp dữ liệu
df_combined = df_main.merge(df_articles, on="id", how="left")

# 4. DỌN DẸP CỘT (Quan trọng nhất để sửa lỗi KeyError)
# Ưu tiên lấy cột từ file x, nếu không có thì lấy file y, rồi đổi tên về chuẩn
if "snopes_url_x" in df_combined.columns:
    df_combined["snopes_url"] = df_combined["snopes_url_x"]
elif "snopes_url_y" in df_combined.columns:
    df_combined["snopes_url"] = df_combined["snopes_url_y"]

# Loại bỏ tất cả các cột rác có đuôi _x, _y và Unnamed
cols_to_keep = [c for c in df_combined.columns if not (c.endswith('_x') or c.endswith('_y') or 'Unnamed' in c)]
df_combined = df_combined[cols_to_keep]

# 5. LƯU FILE
output_path = os.path.join(current_dir, "verite_combined.csv")
df_combined.to_csv(output_path, index=False)

print(f"✅ Đã tạo file thành công: {output_path}")
print(f"📊 Cấu trúc cột mới: {df_combined.columns.tolist()}")