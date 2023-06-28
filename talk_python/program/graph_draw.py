# グラフ描画
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

fig = make_subplots(rows=5, cols=1, subplot_titles=['2D Probability Distribution'])
df = pd.read_csv("./data/child/likelihood_distribution/action_0.csv", sep=" ", header=None)
print(df[0].to_numpy().reshape([5, -1], order="F"))

# fig.add_trace(
#     go.Heatmap(z = df[0].to_numpy().reshape([5, -1], order="F")),
#     row=1, col=1
# )


# fig.show()

plt.imshow(df[0].to_numpy().reshape([5, -1], order="F"))
plt.show()