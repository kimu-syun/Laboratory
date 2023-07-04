import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# フィギュア1の設定
fig1, ax1 = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
ax1.plot(x, y)
ax1.set_title('Figure 1')

# フィギュア2の設定
fig2, ax2 = plt.subplots()
ax2.plot(x, -y)
ax2.set_title('Figure 2')

# ボタンクリック時のコールバック関数
def on_button_clicked(event):
    # クリックされたボタンによって表示を切り替える
    if event.inaxes == ax1_button:
        fig2.canvas.manager.window.hide()
        fig1.canvas.manager.window.show()
    elif event.inaxes == ax2_button:
        fig1.canvas.manager.window.hide()
        fig2.canvas.manager.window.show()

# ボタンの設定
ax1_button = fig1.add_axes([0.7, 0.05, 0.1, 0.075])
ax1_button_obj = Button(ax1_button, 'Figure 1')
ax1_button_obj.on_clicked(on_button_clicked)

ax2_button = fig2.add_axes([0.7, 0.05, 0.1, 0.075])
ax2_button_obj = Button(ax2_button, 'Figure 2')
ax2_button_obj.on_clicked(on_button_clicked)

# グラフをキャンバスに描画
canvas = FigureCanvas(fig1)
canvas.draw()

# グラフをHTMLファイルに保存
html_file = 'output.html'
canvas.print_html(html_file)

print(f"HTMLファイル '{html_file}' に保存しました。")
