# -*- coding: utf-8 -*-
"""
融合流程：
1. SVM   ：分辨 target_ids(內圈) vs 其他
2. MLP   ：只對內圈再分 1/2/3
3. 整合  ：final_pred = 0(外圈) | 1/2/3 (內圈細類)
4. 輸出  ：ckpt & 整張 CSV
"""

import pathlib, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# ---------- 0. 讀資料 ----------
df = pd.read_csv("6d_data.csv", dtype={"ID": str})

# ---------- 1. 參數 ----------
target_ids = {"1", "2", "3"}          # 內圈欲細分的 ID
feat_cols  = ["S1", "S2", "S3"]       # 共用特徵

# ---------- 2. 建二元 label (SVM 用) ----------
df["label_bin"] = df["ID"].isin(target_ids).astype(int)   # 1 = 內圈

X = df[feat_cols].values
y_bin = df["label_bin"].values

# ---------- 3. 切 train / test ----------
X_tr, X_te, y_tr_bin, y_te_bin, idx_tr, idx_te = train_test_split(
    X, y_bin, np.arange(len(df)), test_size=0.2, stratify=y_bin, random_state=42
)

# ---------- 4. 單一 scaler 先 fit ----------
scaler = StandardScaler().fit(X_tr)
X_tr_std = scaler.transform(X_tr)
X_te_std = scaler.transform(X_te)

# ---------- 5. 訓練 SVM ----------
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm.fit(X_tr_std, y_tr_bin)

print("\n=== SVM 評估 (外圈 vs 內圈) ===")
svm_pred_te = svm.predict(X_te_std)
print(classification_report(y_te_bin, svm_pred_te, digits=3))

# ---------- 6. 為 MLP 準備資料 ----------
# 只取「真實 ID 在 target_ids」且 SVM 預測為 1 的樣本做訓練
mask_tr_inner = (y_tr_bin == 1) & (svm.predict(X_tr_std) == 1)
mask_te_inner = (y_te_bin == 1) & (svm_pred_te == 1)

X_tr_mlp = X_tr_std[mask_tr_inner]
y_tr_mlp = df.loc[idx_tr[mask_tr_inner], "ID"].values          # '1','2','3'

X_te_mlp = X_te_std[mask_te_inner]
y_te_mlp = df.loc[idx_te[mask_te_inner], "ID"].values

# ---------- 7. 訓練 MLP ----------
mlp = MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation="relu",
    solver="adam",
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=500,
    alpha=1e-4,
    random_state=42
).fit(X_tr_mlp, y_tr_mlp)

print("\n=== MLP 評估 (ID 1/2/3 細分) ===")
mlp_pred_te = mlp.predict(X_te_mlp)
print(classification_report(y_te_mlp, mlp_pred_te, digits=3))
print("Accuracy (inner-test):", accuracy_score(y_te_mlp, mlp_pred_te))

# ---------- 8. 推論全資料 ----------
X_std_full        = scaler.transform(X)
df["pred_svm"]    = svm.predict(X_std_full)              # 0/1
df["pred_mlp"]    = np.nan
inner_mask_full   = df["pred_svm"] == 1
df.loc[inner_mask_full, "pred_mlp"] = mlp.predict(X_std_full[inner_mask_full])

# 整合欄位：外圈→0，內圈→其對應 ID
df["final_pred"] = np.where(df["pred_svm"]==0, "0", df["pred_mlp"])

# ---------- 9. CKPT ----------
ckpt_dir = pathlib.Path("ckpt"); ckpt_dir.mkdir(exist_ok=True, parents=True)
joblib.dump(scaler, ckpt_dir / "scaler.pkl")
joblib.dump(svm,    ckpt_dir / "svm.pkl")
joblib.dump(mlp,    ckpt_dir / "mlp.pkl")
print("✓ scaler / svm / mlp saved to ckpt/")

# ---------- 10. 可視化 (簡單示例) ----------
# cmap = {"0":"gray", "1":"gold", "2":"limegreen", "3":"deepskyblue"}
# colors = df["final_pred"].map(cmap).to_numpy()

# fig = plt.figure(figsize=(6,6))
# ax  = fig.add_subplot(111, projection="3d")
# ax.scatter(df["S1"], df["S2"], df["S3"], c=colors, s=3, alpha=0.75)
# ax.set(xlabel="S1", ylabel="S2", zlabel="S3", title="Final prediction (gray = outer)")
# plt.tight_layout(); plt.savefig("final_pred_3d.png", dpi=200); plt.close()
# print("✓ final_pred_3d.png saved")
