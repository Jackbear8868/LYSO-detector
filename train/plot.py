import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 讀取資料
df = pd.read_csv('6d_data.csv')
df['ID'] = df['ID'].astype(str)  # 保證分類正確

# 維度對應組合
dimension_pairs = [('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4'), ('S4', 'S5'), ('S5', 'S6')]

# 建立色盤（hls 支援 >30 類別）
unique_ids = df['ID'].unique()
num_classes = len(unique_ids)
palette = sns.color_palette("hls", num_classes)
id_to_color = {id_: palette[i] for i, id_ in enumerate(unique_ids)}

# 建立儲存資料夾
os.makedirs("./plots", exist_ok=True)

# 為每組維度畫圖 + 儲存
for (x_dim, y_dim) in dimension_pairs:
    plt.figure(figsize=(6, 6))  # 每張圖單獨開
    ax = sns.scatterplot(
        data=df,
        x=x_dim,
        y=y_dim,
        hue='ID',
        palette=id_to_color,
        s=2,
        linewidth=0,
        alpha=0.6
    )
    ax.set_title(f'{x_dim} vs {y_dim}')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    
    # 顯示圖例（放在外面右上）
    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='x-small', title='ID', ncol=1)
    ax.legend_.remove()

    plt.tight_layout()
    plt.savefig(f"./plots/{x_dim}_vs_{y_dim}.png", dpi=300, bbox_inches='tight')
    plt.close()  # 關閉當前圖，避免重疊
