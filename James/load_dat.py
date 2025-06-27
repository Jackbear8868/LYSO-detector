import numpy as np
import pandas as pd

with open("clusterID.dat", "rb") as f:
    clusterID = np.load(f)

# 建立 DataFrame，並指定「ID」為欄位名稱
df = pd.DataFrame({"ID": clusterID})

# 把索引改成從 1 開始，並命名為「Event」
df.index = np.arange(1, len(df) + 1)
df.index.name = "Event"

# 輸出成 CSV
df.to_csv("clusterID.csv")
