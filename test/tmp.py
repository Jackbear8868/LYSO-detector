# -*- coding: utf-8 -*-
import pathlib, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def predict(df, axis):
    X       = df[axis].values
    X_std   = scaler.transform(X)
    df["pred"] = svm_rbf.predict(X_std)          # 0 / 1

    # ---------- 4. 3D 視覺化 ----------
    colors = np.where(df["pred"] == 1, "crimson", "lightgray")
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(df[axis[0]], df[axis[1]], df[axis[2]], c=colors, s=2, alpha=0.7)
    ax.set(xlim=(0,1000), ylim=(0,1000), zlim=(0,1000),
        xlabel=axis[0], ylabel=axis[1], zlabel=axis[2],
        title="RBF-SVM prediction (red = pred 1)")
    plt.tight_layout()
    plt.savefig(f"classified_{axis[0]}_{axis[1]}_{axis[2]}_ckpt.png", dpi=200)
    plt.close()
    print(f"✓ classified_{axis[0]}_{axis[1]}_{axis[2]}_ckpt.png saved")


# ---------- 0. 讀檔 ----------
CSV_PATH = "6d_data.csv"
df = pd.read_csv(CSV_PATH, dtype={"ID": str})

# ---------- 1. 載入 ckpt ----------
CKPT_DIR = pathlib.Path("svm_ckpt")
svm_rbf = joblib.load(CKPT_DIR / "svm_outsider.pkl")   # ← 你的模型
scaler  = joblib.load(CKPT_DIR / "scaler_outsider.pkl")  # ← 配套 scaler
print("✓ ckpt loaded")

# ---------- 2. 產生特徵並做預測 ----------
predict(df, ["S1", "S2", "S3"])
predict(df, ["S6", "S5", "S4"])