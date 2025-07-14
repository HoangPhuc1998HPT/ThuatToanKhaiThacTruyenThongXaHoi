import pandas as pd
from ucimlrepo import fetch_ucirepo


# Bước 1: Tải bộ dữ liệu
data = fetch_ucirepo(id=27)  # Credit Approval
df = pd.concat([data.data.features, data.data.targets], axis=1)

# Bước 2: Loại bỏ các dòng chứa dấu '?'
df = df[~df.isin(['?']).any(axis=1)].reset_index(drop=True)

# Bước 3: Giữ các cột phân loại (danh mục) + quyết định
cat_columns = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
df_filtered = df[cat_columns + ['A16']].copy()

# Bước 4: Đặt cột ID và đổi tên quyết định
df_filtered.insert(0, 'ID', ['O' + str(i+1) for i in range(len(df_filtered))])
df_filtered = df_filtered.rename(columns={'A16': 'Quyết định'})

# Bước 5: Xuất ra file CSV chuẩn cho GUI
df_filtered.to_csv("credit_approval_dataset_csv.csv", index=False, sep=';', encoding='utf-8-sig')

