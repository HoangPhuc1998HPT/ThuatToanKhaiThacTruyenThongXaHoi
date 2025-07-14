import pandas as pd
import random

# Bước 1: Đọc dữ liệu gốc
df = pd.read_csv("tap_tho_csv.csv", sep=';')  # Đảm bảo là file có tiếng Việt chuẩn

# Bước 2: Tách cột ID và phần điều kiện + quyết định
columns = df.columns.tolist()
id_col = columns[0]
other_cols = columns[1:]

# Lấy giá trị duy nhất từ mỗi cột (trừ ID)
unique_values = {col: df[col].dropna().unique().tolist() for col in other_cols}

# Bước 3: Sinh dữ liệu ngẫu nhiên cho các cột còn lại
synthetic_data = []
for i in range(64):
    row = [f"O{i+1}"]  # ID tăng dần: O1, O2, ...
    row += [random.choice(unique_values[col]) for col in other_cols]
    synthetic_data.append(row)

# Bước 4: Tạo DataFrame mới
df_new = pd.DataFrame(synthetic_data, columns=columns)

# Bước 5: Xuất ra CSV với UTF-8-BOM để mở Excel không lỗi tiếng Việt
df_new.to_csv("du_lieu_64dong_tho.csv", index=False, sep=';', encoding='utf-8-sig')

print("✅ Đã tạo xong file du_lieu_10000dong.csv với ID duy nhất và dữ liệu ngẫu nhiên tiếng Việt chuẩn.")
