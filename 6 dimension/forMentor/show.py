import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

# 假設 CSV 檔案名稱為 "data.csv"，請根據實際情況修改
csv_path = "data.csv"

# 讀取 CSV
df = pd.read_csv(csv_path)

# 選取所有 S 開頭的欄位（假設為 S1, S2, S3, ...）
s_columns = [col for col in df.columns if col.startswith('S')]
s_columns.sort()  # 確保順序為 S1, S2, S3,...

unique_labels = sorted(df["ID"].unique())
n = len(unique_labels)
label_to_index = {label: i for i, label in enumerate(unique_labels)}

# 2. 產生高對比顏色盤（這裡使用 tab20，並將 index 打亂使相鄰色差大）
palette = sns.color_palette("husl", n)
permuted_indices = np.linspace(0, 1, n)**0.5  # 非線性 spacing 增加色差

spread_order = [(i * 5) % n for i in range(n)]
shuffled_palette = [palette[i] for i in spread_order]
cmap = mcolors.ListedColormap(shuffled_palette)


# 3. 用新的顏色順序對應到每個 ID（返回 RGB 值陣列）
colors = df["ID"].map(label_to_index)

# 依照相鄰的兩個欄位畫出 2D 散點圖
output_dir = "colored_plots"
os.makedirs(output_dir, exist_ok=True)

sgmin = 0.
sgmax = 2000.
nbin = 200       

for i in range(len(s_columns) - 1):
    x_col = s_columns[i]
    y_col = s_columns[i + 1]
    
    plt.figure(20)
    plt.scatter(df[x_col], df[y_col], c=colors, cmap=cmap,marker='.',s=2,linewidths=0)
    plt.xlim([sgmin,sgmax])
    plt.ylim([sgmin,sgmax])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.grid(True)
    filename = f"{x_col}_vs_{y_col}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()