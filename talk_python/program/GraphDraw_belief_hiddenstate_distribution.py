# グラフ描画 (q(y|x,a))
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go




def graph_draw(action_range, agent, epoch):

    for t in range(0, epoch):

        fig = []

        fig = make_subplots(
            rows=(action_range+1)//2,
            cols=2,
            subplot_titles=[f"$a = {i}$" for i in range(0, action_range)]
            )


        df = pd.read_csv(f"./data/{agent}/belief_hiddenstate_distribution/epoch_{t}.csv", sep=" ", header=None)
        df = df.T


        for i in range(0, action_range):
            fig.add_trace(go.Heatmap(
                z = df[i].to_numpy().reshape([action_range, -1]),
                x=(1,2),
                y=(1,2),
                zmin = 0,
                zmax = 1.0,
                opacity = 0.5,
            ), row=(i//2)+1, col=(i%2)+1)
        fig.update_layout(
            title = f"${agent}-epoch{t}-q(x|a)-$",
            )
        fig.update_xaxes(linecolor='black', gridcolor='black',mirror=True, title = dict(text="emotion"))
        fig.update_yaxes(linecolor='black', gridcolor='black',mirror=True, title = dict(text="relation"))


        fig.write_image(f"./graph/{agent}/belief_hiddenstate_distribution/epoch_{t}.png")
        fig.show()


