import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

X = np.arange(start=-0.5, stop=4.5, step=0.1)

# pdfで確率密度関数を生成
norm_pdf = stats.norm.pdf(x=X, loc=4, scale=0.8) # 期待値=4, 標準偏差=0.8

# 可視化
plt.plot(X, norm_pdf)
plt.xlabel("確率変数X", fontsize=13)
plt.ylabel("確率密度pdf", fontsize=13)
# plt.show()
print(round(stats.norm.pdf(x=2, loc=4, scale=0.8), 4))