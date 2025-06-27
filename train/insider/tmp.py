import pathlib, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
def plot_3d(df, axis, colors, fname, title):
    """繪製 3D 散點圖"""
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(df[axis[0]], df[axis[1]], df[axis[2]],
               c=colors, s=3, alpha=0.75)
    ax.set(xlabel=axis[0], ylabel=axis[1], zlabel=axis[2],
           title=title, xlim=(0, 2000), ylim=(0, 2000), zlim=(0, 2000))
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()
    print(f"✓ {fname} saved")

# ------------------------------------------------------------
def main():
    csv_path  = "merge.csv"
    ckpt_dir  = pathlib.Path("ckpt")
    feat_cols = ["S1", "S2", "S3"]  # ← 測試不同視窗時這裡可以換

    # ✅ SVM 輸出 → 真實 ID 的對應表（自行指定）
    label_map = {0: 0, 4: 4, 5: 5, 6: 6}

    # ---------- 1. 讀資料 ----------
    df = pd.read_csv(csv_path)
    df = df[df["pred"] == 0].reset_index(drop=True)

    # ---------- 2. 載 ckpt ----------
    scaler = joblib.load(ckpt_dir / "scaler.pkl")
    model  = joblib.load(ckpt_dir / "svm.pkl")

    # ---------- 3. 準備 X 與 y_true ----------
    X  = df[feat_cols].values
    Xs = scaler.transform(X)

    # ✅ 自動從 label_map 取出「內圈」真實 ID
    inner_ids = set(label_map.values())
    y_true = df["ID"].astype(int).apply(lambda x: x if x in inner_ids else 0).to_numpy()

    # ---------- 4. 預測 & 對映回真實 ID ----------
    y_pred_raw = model.predict(Xs)
    y_pred     = np.vectorize(label_map.get)(y_pred_raw)
    df["pred_cls"] = y_pred

    # ---------- 5. 評估 ----------
    # acc = accuracy_score(y_true, y_pred)
    # print(f"\n=== Overall accuracy : {acc:.4f} ===\n")
    # print(classification_report(y_true, y_pred, digits=3))

    # ---------- 6. 繪圖（僅限 3 維） ----------
    cmap = {0:"gray", 4:"gold", 5:"limegreen", 6:"deepskyblue"}
    colors = df["pred_cls"].map(cmap).fillna("red").to_numpy()
    plot_3d(
        df, feat_cols, colors,
        fname=f"{feat_cols[0]}_{feat_cols[1]}_{feat_cols[2]}.png",
        title="SVM final prediction (mapped to 0/22/21/20)"
    )

if __name__ == "__main__":
    main()
