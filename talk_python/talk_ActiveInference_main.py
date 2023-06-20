# 能動的推論を使った会話モデル

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib
from math import log
import random
import pandas as pd

# 変数設定(x,y,a)する構造体
class agent:

    def __init__(self):
        self.hidden_state = np.empty(2)
        self.sensory_signal = np.arange(sensory_range)
        self.action = np.arange(action_range)

#二次元正規分布の確率密度を返す関数
def gaussian(x, mu, sigma):
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = 2
    return np.exp(- ((x - mu) @ inv @ (x - mu).T) / 2.0) / (np.sqrt((2 * np.pi) ** n * det))


# x,y,aの程度
emotion_range = 5
relation_range = 5
sensory_range = 5
action_range = 5
hidden_state_range = emotion_range * relation_range


# 真の値
child = agent()
parent = agent()

# 初期値設定(x)
child.hidden_state = np.array([0, 0])
parent.hidden_state = np.array([0, 0])

# 分布の設定
likelihood_distribution = np.zeros((action_range, hidden_state_range, sensory_range)) # p(y|x,a)
belief_hiddenstate_distribution = np.zeros((action_range, hidden_state_range)) # q(x|a)
preference_distribution = np.zeros((sensory_range)) # p~(y)


# 尤度分布作成 p(y|x,a)
for a in range(0, action_range):
    for x in range(0, hidden_state_range):
        for y in range(0, sensory_range):
            likelihood_distribution[a, x, y] = stats.norm.pdf(x = y, loc = a + (x % 5) / float(8), scale = 1.0)
        # print(a, x, '期待値 :', a + (x % 5) / float(8),likelihood_distribution[a, x])
        # 尤度分布をcsv保存
        np.savetxt(f"./data/likelihood_distribution/action_{a}.csv", likelihood_distribution[a], fmt="%.5f")

# 信念分布作成 q(x|a)
for a in range(0, action_range):
    belief_distribution_mu = np.array([a, a])
    belief_distribution_sigma = np.array([[0.4, 0], [0, 0.4]])
    for x in range(0, hidden_state_range):
        r = x % relation_range
        e = (x - r) // relation_range
        belief_hiddenstate_distribution[a, x] = gaussian(np.array([e, r]), belief_distribution_mu, belief_distribution_sigma)


# 選好の分布作成 p~(y)
preference_distribution = np.array([0.1, 0.2, 0.4, 0.5, 0.8])



# active inference
count = 10
F_expected = np.zeros((count, action_range)) # 期待自由エネルギーを保存
epistemic_value_save = np.zeros((count, action_range))
predicted_surprised_save = np.zeros((count, action_range))
action = -1 # actionを保存

for i in range(0, count):
    print(f"{i+1}回目")
    # y_signal = int(input("感覚信号の入力 : "))
    y_signal = random.randint(0, 4)
    print(f"感覚信号 : {y_signal}")


    # 感覚信号の信念分布 q(y|a)
    belief_sensory_distribution = np.zeros((action_range, sensory_range)) # q(y|a)

    # 隠れ状態の条件付確率分布 q(x|y,a)
    belief_conditional_hiddenstate_distribution = np.zeros((action_range, hidden_state_range, sensory_range)) # q(x|y,a)

    # q(y|a)
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                belief_sensory_distribution[a, y] = \
                    belief_sensory_distribution[a, y] + (likelihood_distribution[a, x, y] * belief_hiddenstate_distribution[a, x]) # q(y|a) = Σ_x{p(y|x)*q(x)}
        np.savetxt(f"./data/belief_sensory_distribution/count_{i}_action_{a}.csv", belief_sensory_distribution[a], fmt="%.5f")
        np.savetxt(f"./data/belief_hiddenstate_distribution/count_{i}_action_{a}.csv", belief_hiddenstate_distribution[a], fmt="%.5f")

    # q(x|y,a)
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                belief_conditional_hiddenstate_distribution[a, x, y] = \
                    ((likelihood_distribution[a, x, y] * belief_hiddenstate_distribution[a, x]) / belief_sensory_distribution[a, y]) # q(x|y,a) = {p(y|x,a)q(x|a)}/q(y|a)
        np.savetxt(f"./data/belief_conditional_hiddenstate_distribution/count_{i}_action_{a}.csv", belief_conditional_hiddenstate_distribution[a], fmt="%.5f")


    # epistemic value(エピスティミック価値)信念の更新度合い
    epistemic_value = np.zeros(action_range)

    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                epistemic_value[a] = belief_sensory_distribution[a, y] * belief_conditional_hiddenstate_distribution[a, x, y]\
                      * log((belief_conditional_hiddenstate_distribution[a, x, y]) / belief_hiddenstate_distribution[a, x]) # Σ_y{q(y|a)Σ_x{q(x|y,a)log(q(x|y,a)/q(x|a))}}

    # predicted surprised(予測サプライズ)
    predicted_surprised = np.zeros(action_range)

    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                predicted_surprised[a] = predicted_surprised[a] - belief_sensory_distribution[a, y] * log(preference_distribution[y])

    # 期待自由エネルギーを計算、保存
    for a in range(0, action_range):
        F_expected[i, a] = -epistemic_value[a] + predicted_surprised[a]
        epistemic_value_save[i, a] = epistemic_value[a]
        predicted_surprised_save[i, a] = predicted_surprised[a]

    # q(x)を更新 q(x|y,a)->q(x)
    for a in range(0, action_range):
        for x in range(0, hidden_state_range):
            belief_hiddenstate_distribution[a, x] = belief_conditional_hiddenstate_distribution[a, x, y_signal]

    print(f"FE{F_expected}")

np.savetxt(f"./data/F_expected.csv", F_expected, fmt="%.5f")
np.savetxt(f"./data/pistemic_value.csv", epistemic_value_save, fmt="%.5f")
np.savetxt(f"./data/predicted_surprised.csv", predicted_surprised_save, fmt="%.5f")
