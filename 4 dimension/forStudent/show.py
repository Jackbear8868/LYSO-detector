import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations

# 讀取檔案
df = pd.read_csv("label.csv")

# 選擇 S 開頭欄位
s_columns = [col for col in df.columns if col.startswith("S")]
s_columns.sort()

# ID → 顏色索引
unique_labels = sorted(df["ID"].unique())
label_to_index = {label: i for i, label in enumerate(unique_labels)}
colors = df["ID"].map(label_to_index)

label_counts = df["ID"].value_counts().sort_index()

# 建立輸出資料夾
output_dir = "colored_plots"
os.makedirs(output_dir, exist_ok=True)

# 對每一組 S 特徵組合畫圖（S1 vs S2, S1 vs S3, ..., S3 vs S4）
for x_col, y_col in combinations(s_columns, 2):
    plt.figure(figsize=(6, 6))
    plt.scatter(df[x_col], df[y_col], c=colors, cmap="tab20", s=2, alpha=0.8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.grid(True)
    
    filename = f"{x_col}_vs_{y_col}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
