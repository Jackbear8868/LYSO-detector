import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib, pathlib

def predict_one(df, axis, scaler, svm, mlp):

    X_std   = scaler.transform(df[axis])
    mask    = svm.predict(X_std) == 1
    pred    = np.full(len(df), np.nan)
    if mask.any():
        pred[mask] = mlp.predict(X_std[mask])
    return pred        # ndarray(float 或 int)

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


def outsider(df):
    ckpt = pathlib.Path("ckpt")
    # ---------- 1. 讀 ckpt ----------
    svm   = joblib.load(ckpt / "svm.pkl")
    mlp   = joblib.load(ckpt / "mlp.pkl")
    scaler= joblib.load(ckpt / "scaler.pkl")
    # ---------- 2. 兩次分類 ----------
    pred_run1 = predict_one(df, ["S1","S2","S3"], scaler, svm, mlp)
    pred_run2 = predict_one(df, ["S4","S3","S2"], scaler, svm, mlp)
    # ---------- 3. 合併唯一欄位 ----------
    df = df.copy()
    df["pred"] = "0"          # 先全部填 0（字串）
    # ⇣ 依照需求把不同 run 映射成「3 位數 ID」
    map1 = {1:"1", 2:"2", 3:"3"}
    map2 = {1:"17", 2:"16", 3:"15"}

    # Run-1 結果寫進去
    mask1 = ~np.isnan(pred_run1)
    df.loc[mask1, "pred"] = pd.Series(pred_run1[mask1]).map(map1).values

    # Run-2 結果；若與 run-1 衝突，這裡會覆蓋
    mask2 = ~np.isnan(pred_run2)
    df.loc[mask2, "pred"] = pd.Series(pred_run2[mask2]).map(map2).values

    cmap = {"0":"gray", "1":"gold", "2":"limegreen", "3":"deepskyblue","17":"orchid", "16":"cyan", "15":"magenta"}
    colors = df["pred"].map(cmap).to_numpy()

    # ---------- 4.（可選）視覺化最終結果 ----------
    plot_3d(df, ["S1","S2","S3"], colors, "final_3d_s1-s3.png", "Merged result (gray=unclassified)")
    plot_3d(df, ["S4","S3","S2"], colors, "final_3d_s6-s4.png", "Merged result (gray=unclassified)")

    return df


# ---------- 0. 讀資料 ----------
df = pd.read_csv("4d_data.csv", dtype={"ID": str})

df = outsider(df)
df.to_csv("4d_result.csv", index=False)

# ---------- 2. 指定要看的 ID ----------
# target_ids = ["1", "2", "3", "23", "24", "25"]          # ← 你要評估哪幾種就填哪幾個

# sub = df[df["ID"].isin(target_ids)].copy()

# # ---------- 3. 計算 Accuracy ----------
# acc = (sub["pred"] == sub["ID"]).mean()        # .mean() = 正確率

# print(f"Accuracy for IDs {target_ids}: {acc:.3f}")
