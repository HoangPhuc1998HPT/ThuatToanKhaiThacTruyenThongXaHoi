import pandas as pd
from ucimlrepo import fetch_ucirepo

# Bước 1: Tải bộ dữ liệu
data = fetch_ucirepo(id=184)  # Acute Inflammations
df_raw = pd.concat([data.data.features, data.data.targets], axis=1)

# Bước 2: Trộn 2 cột quyết định thành 1
def combine_decision(row):
    if row["bladder-inflammation"] == "yes" and row["nephritis"] == "yes":
        return "bladder-inflammation and nephritis"
    elif row["bladder-inflammation"] == "yes":
        return "bladder-inflammation"
    elif row["nephritis"] == "yes":
        return "nephritis"
    else:
        return "none"

df_raw["decision"] = df_raw.apply(combine_decision, axis=1)

# Bước 3: Tạo cột ID tuần tự
df_raw.insert(0, 'ID', [f'O{i+1}' for i in range(len(df_raw))])

# Bước 4: Giữ lại các cột cần thiết: ID , cột điều kiện, cột quyết định đã gộp
df = df_raw[['ID'] + data.data.features.columns.tolist() + ['decision']]

# Xuất file CSV
df.to_csv('acute_inflammations_dataset_csv.csv', index=False, sep=';', encoding='utf-8-sig')
