# グラフ描画 (p(y|x,a))
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def graph_draw(y_range, action_range, agent):


    for a in range(0, action_range):
        # データ入力
        df = pd.read_csv(f"./data/{agent}/likelihood_distribution/action_{a}.csv", sep=" ", header=None)


        fig = make_subplots(
            rows=(y_range+1)//2,
            cols=2,
            subplot_titles=[f"$y = {i}$" for i in range(0, y_range)]
            )



        for i in range(0, y_range):
            fig.add_trace(go.Heatmap(
                z = df[i].to_numpy().reshape([y_range, -1]),
                x=(1,2),
                y=(1,2),
                zmin = 0,
                zmax = 1.0,
                opacity = 0.5,
            ), row=(i//2)+1, col=(i%2)+1)


        fig.update_layout(
            title = f"${agent}-action{a}-p(y|x,a)-$",
            )


        fig.update_xaxes(linecolor='black', gridcolor='black',mirror=True, title = dict(text="emotion"))
        fig.update_yaxes(linecolor='black', gridcolor='black',mirror=True, title = dict(text="relation"))

        fig.show()

        fig.write_html(f"./graph/{agent}/likelihood_distribution/action_{a}.html")