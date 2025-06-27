# -*- coding: utf-8 -*-
"""
SVM 多類別分類流程：
將 ID = 4, 5, 6 分為獨立類別，其餘 ID 合併為 0 類（外圈）
"""
import matplotlib.pyplot as plt
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def plot_3d(df_sub, axis, colors, fname, title):
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(df_sub[axis[0]], df_sub[axis[1]], df_sub[axis[2]],
               c=colors, s=2, alpha=0.75)
    ax.set(xlabel=axis[0], ylabel=axis[1], zlabel=axis[2], title=title, xlim=(0, 2000), ylim=(0, 2000), zlim=(0, 2000))
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print("✓", fname, "saved")

# ---------- 0. 讀資料 ----------
df = pd.read_csv("6d_data.csv")

# 僅處理 pred == 0 的資料
# df = df[df["pred"] == 0].reset_index(drop=True)

# ---------- 1. 準備分類資料 ----------
feat_cols = ["S1", "S2", "S3"]
X = df[feat_cols].values
y = df["ID"].astype(str)  # 轉成文字方便處理

# 將 ID 映射為分類類別：4/5/6 保留，其餘為 0
target_ids = {"4", "5", "6"}
y_mapped = np.array([int(i) if i in target_ids else 0 for i in y])
df["mapped_id"] = y_mapped  # 保留以利繪圖

# ---------- 2. 切分訓練 / 測試 ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_mapped, test_size=0.2, stratify=y_mapped, random_state=42
)

# ---------- 3. 標準化 ----------
scaler = StandardScaler().fit(X_tr)
X_tr_std = scaler.transform(X_tr)
X_te_std = scaler.transform(X_te)

# ---------- 4. 訓練 SVM (多類別分類) ----------
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    decision_function_shape="ovo",
    class_weight="balanced",  # 解決類別不平衡
    probability=True,
    random_state=42
)
svm.fit(X_tr_std, y_tr)

# ---------- 5. 評估 ----------
y_pred = svm.predict(X_te_std)
acc = accuracy_score(y_te, y_pred)
report = classification_report(y_te, y_pred, digits=3)

print(f"\n=== 準確率 (Accuracy): {acc:.3f} ===\n")
print("=== 分類報告 ===")
print(report)

# ---------- 6. 模型儲存 ----------
ckpt_dir = pathlib.Path("ckpt"); ckpt_dir.mkdir(exist_ok=True, parents=True)
joblib.dump(scaler, ckpt_dir / "scaler.pkl")
joblib.dump(svm,    ckpt_dir / "svm.pkl")
print("✓ 模型與標準化器已儲存至 ckpt/ 資料夾")

# ---------- 7. 畫 3d 圖 ----------
X_std_full = scaler.transform(X)
df["final_pred"] = svm.predict(X_std_full)

final_acc = accuracy_score(df["mapped_id"], df["final_pred"])
print(f"\n=== Final prediction accuracy (on all pred==0 rows): {final_acc:.3f} ===")

# 定義顏色映射
cmap = {
    0: "gray",
    4: "gold",
    5: "limegreen",
    6: "deepskyblue"
}
colors = df["final_pred"].map(cmap).to_numpy()

# ---------- 9. 呼叫繪圖 ----------
plot_3d(df, axis=["S1", "S2", "S3"], colors=colors,fname="train_result.png", title="Final prediction (gray = outer)")
