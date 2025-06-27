import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("data.csv")

# 第一組: S1-S2-S3
trace1 = go.Scatter3d(
    x=df["s1"], y=df["s2"], z=df["s3"],
    mode="markers",
    marker=dict(size=2, color="blue"),
    name="S1–S2–S3"
)

# 第二組: S2-S3-S4
trace2 = go.Scatter3d(
    x=df["s2"], y=df["s3"], z=df["s4"],
    mode="markers",
    marker=dict(size=2, color="red"),
    name="S2–S3–S4"
)

# 合併圖層
fig = go.Figure(data=[ trace2])

fig.update_layout(
    scene=dict(
        xaxis=dict(title='X', range=[0, 2000]),
        yaxis=dict(title='Y', range=[0, 2000]),
        zaxis=dict(title='Z', range=[0, 2000])
    ),
    title="Overlay of (S1,S2,S3) and (S2,S3,S4)",
    legend=dict(x=0, y=1)
)

fig.show()
