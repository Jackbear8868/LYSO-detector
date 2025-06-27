import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib, pathlib

class LYSO_detector:
    def __init__(self, df,n, ckpt_dir):   # 建構子 (constructor)
        # 實例變數 (attributes)
        self.n = n
        self.df = df
        # ---------- 讀 ckpt ----------
        ckpt = pathlib.Path(ckpt_dir)
        self.scaler = joblib.load(ckpt / "scaler.pkl")
        self.outside_svm = joblib.load(ckpt / "outside_svm.pkl")
        self.inside_svm  = joblib.load(ckpt / "inside_svm.pkl")
        self.mlp = joblib.load(ckpt / "mlp.pkl")
    
    def predict_outside_and_mlp(self, df, axis):
        X_std   = self.scaler.transform(df[axis])
        mask    = self.outside_svm.predict(X_std) == 1
        pred    = np.full(len(df), np.nan)
        if mask.any():
            pred[mask] = self.mlp.predict(X_std[mask])
        return pred        # ndarray(float 或 int)

    
    def predict(self):                  # 方法 (method)
        pred1 = self.predict_outside_and_mlp(self.df, ["S1","S2","S3"])
        pred2 = self.predict_outside_and_mlp(self.df, [f"S{self.n}", f"S{self.n-1}", f"S{self.n-2}"])
        
        self.df["pred"] = "0"          # 先全部填 0（字串）
        # ⇣ 依照需求把不同 run 映射成「3 位數 ID」
        map1 = {1:"1", 2:"2", 3:"3"}
        map2 = {1:f"{4*self.n+1}", 2:f"{4*self.n}", 3:f"{4*self.n-1}"}

        # Run-1 結果寫進去
        mask1 = ~np.isnan(pred1)
        self.df.loc[mask1, "pred"] = pd.Series(pred1[mask1]).map(map1).values

        # Run-2 結果；若與 run-1 衝突，這裡會覆蓋
        mask2 = ~np.isnan(pred2)
        self.df.loc[mask2, "pred"] = pd.Series(pred2[mask2]).map(map2).values

