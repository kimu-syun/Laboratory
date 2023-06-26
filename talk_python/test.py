import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 各時間ステップごとのデータを生成する関数（例としてランダムなデータを使用）
def generate_data():
    return np.random.rand(100, 100)  # 100x100のランダムなデータを生成

# 時間のステップ数を指定
time_steps = 10

# サブプロットを作成
fig = make_subplots(rows=1, cols=1, subplot_titles=['2D Probability Distribution'])

# ヒートマップを更新するための関数
def update_heatmap(t):
    data = generate_data()  # データを取得または生成

    fig.data = []  # 既存のデータをクリア

    # ヒートマップを追加
    fig.add_trace(
        go.Heatmap(
            z=data,
            colorscale='hot',
            zmin=0,
            zmax=1,
            showscale=(t == 0),  # 条件式により True/False を指定
            zauto=False,
            zmid=0.5,
            name='2D Probability Distribution'
        ),
        row=1, col=1
    )

    # レイアウトを設定
    fig.update_layout(
        title=f'2D Probability Distribution - Time Step {t}',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        coloraxis=dict(colorbar=dict(title='Density')),
        showlegend=False
    )

# 初回のヒートマップを描画
update_heatmap(0)


# ボタンを作成して時間の切り替えを制御
buttons = []
for t in range(time_steps):
    button = dict(
        label=f'Time Step {t}',
        method='update',
        args=[{'visible': [False] * time_steps}, {'title': f'2D Probability Distribution - Time Step {t}'}]
    )
    button['args'][0]['visible'][t] = True
    button['args'][0]['visible'][0] = True  # 初回は可視化
    button['args'][1]['title'] = f'2D Probability Distribution - Time Step {t}'
    buttons.append(button)

# レイアウトにボタンを追加
fig.update_layout(updatemenus=[dict(buttons=buttons, direction='down', showactive=True)])

# グラフを表示
fig.show()
