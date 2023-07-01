# グラフ描画
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

time_steps = 10


# データ入力
df = pd.read_csv("./data/child/likelihood_distribution/action_0.csv", sep=" ", header=None)
#print(df[0].to_numpy().reshape([5, -1], order="F"))



# figure作成
fig, axes = plt.subplots(3, 2, tight_layout=True, figsize = (8, 10))
fig.suptitle("尤度分布 $p(y|x,a)$")
fig.delaxes(axes[2, 1])


for i in range(0, 5):
    axes[i//2][i%2].set_xlabel("emotion")
    axes[i//2][i%2].set_ylabel("relation")
    axes[i//2][i%2].grid()
    axes[i//2][i%2].set_title(f"$y$ = {i+1}")
    axes[i//2][i%2].set_xlim(left=0, right=4)
    axes[i//2][i%2].set_ylim(bottom=0, top=4)
    heatmap = axes[i//2][i%2].imshow(df[i].to_numpy().reshape([5, -1], order="F"), origin="lower", vmin = 0, vmax = 1)


divider = make_axes_locatable(axes[2,1])
cax = divider.append_axes("bottom", size="5%", pad=0.5) #append_axesで新しいaxesを作成

fig.colorbar(heatmap, cax=cax, orientation="horizontal") #新しく作成したaxesであるcaxを渡す。

plt.show()
