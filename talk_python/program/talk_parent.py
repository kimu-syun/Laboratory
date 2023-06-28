# 親の能動推論
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib
from math import log
import random
import pandas as pd



# x,y,aの範囲
emotion_range = 5
relation_range = 5
sensory_range = 5
action_range = 5
hidden_state_range = emotion_range * relation_range

#二次元正規分布の確率密度を返す関数
def gaussian(x, mu, sigma):
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = 2
    return np.exp(- ((x - mu) @ inv @ (x - mu).T) / 2.0) / (np.sqrt((2 * np.pi) ** n * det))





# 尤度分布作成 p(y|x,a)
def likelihood_distribution_make(likelihood_distribution):
    for a in range(0, action_range):
        for x in range(0, hidden_state_range):
            for y in range(0, sensory_range):
                likelihood_distribution[a, x, y] = stats.norm.pdf(x = y, loc = a + (x % 5) / float(8), scale = 1.0)
            # print(a, x, '期待値 :', a + (x % 5) / float(8),likelihood_distribution[a, x])
            # 尤度分布をcsv保存
        np.savetxt(f"./data/parent/likelihood_distribution/action_{a}.csv", likelihood_distribution[a], fmt="%.5f")
    return likelihood_distribution




# 信念分布作成 q(x|a)
def belief_hiddenstate_distribution_make(belief_hiddenstate_distribution):
    for a in range(0, action_range):
        belief_distribution_mu = np.array([a, a])
        belief_distribution_sigma = np.array([[0.4, 0], [0, 0.4]])
        for x in range(0, hidden_state_range):
            r = x % relation_range
            e = (x - r) // relation_range
            belief_hiddenstate_distribution[a, x] = gaussian(np.array([e, r]), belief_distribution_mu, belief_distribution_sigma)
    return belief_hiddenstate_distribution


# 選好の分布作成 p~(y)
def preference_distribution_make(preference_distribution):
    preference_distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return preference_distribution



# q(y|a)
def belief_sensory_distribution_make(likelihood_distribution, belief_hiddenstate_distribution, belief_sensory_distribution, epoch):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                belief_sensory_distribution[a, y] = \
                    belief_sensory_distribution[a, y] + (likelihood_distribution[a, x, y] * belief_hiddenstate_distribution[a, x]) # q(y|a) = Σ_x{p(y|x)*q(x)}
    np.savetxt(f"./data/parent/belief_sensory_distribution/epoch_{epoch}.csv", belief_sensory_distribution, fmt="%.5f")
    return belief_sensory_distribution


# q(x|y,a)
def belief_conditional_hiddenstate_distribution_make(likelihood_distribution, belief_hiddenstate_distribution, belief_sensory_distribution, belief_conditional_hiddenstate_distribution, epoch):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                belief_conditional_hiddenstate_distribution[a, x, y] = \
                    ((likelihood_distribution[a, x, y] * belief_hiddenstate_distribution[a, x]) / belief_sensory_distribution[a, y]) # q(x|y,a) = {p(y|x,a)q(x|a)}/q(y|a)
        np.savetxt(f"./data/parent/belief_conditional_hiddenstate_distribution/epoch_{epoch}_action_{a}.csv", belief_conditional_hiddenstate_distribution[a], fmt="%.5f")
    return belief_conditional_hiddenstate_distribution


# epistemic value(エピスティミック価値)信念の更新度合い
def epistemic_value_calculate(belief_hiddenstate_distribution, belief_sensory_distribution, belief_conditional_hiddenstate_distribution, epistemic_value, epoch):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                epistemic_value[a] = belief_sensory_distribution[a, y] * belief_conditional_hiddenstate_distribution[a, x, y]\
                      * log((belief_conditional_hiddenstate_distribution[a, x, y]) / belief_hiddenstate_distribution[a, x]) # Σ_y{q(y|a)Σ_x{q(x|y,a)log(q(x|y,a)/q(x|a))}}
    np.savetxt(f"./data/parent/epistemic_value/epoch_{epoch}_epistemic_value.csv", epistemic_value, fmt="%.5f")
    return epistemic_value



# predicted surprised(予測サプライズ)
def predicted_surprised_calculate(belief_sensory_distribution, preference_distribution, predicted_surprised, epoch):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            predicted_surprised[a] = predicted_surprised[a] - belief_sensory_distribution[a, y] * log(preference_distribution[y])
    np.savetxt(f"./data/parent/predicted_surprised/epoch_{epoch}_predicted_surprised.csv", predicted_surprised, fmt="%.5f")
    return predicted_surprised



# 期待自由エネルギーを計算、保存
def F_expected_calculate(epistemic_value, predicted_surprised, F_expected, epoch):
    for a in range(0, action_range):
        F_expected[a] = -epistemic_value[a] + predicted_surprised[a]
    np.savetxt(f"./data/parent/F_expected/epoch_{epoch}_F_expected.csv", F_expected, fmt="%.5f")
    return F_expected


# q(x)を更新 q(x|y,a)->q(x)
def qxUpdate(belief_hiddenstate_distribution, belief_conditional_hiddenstate_distribution, y_signal, a, epoch):
    np.savetxt(f"./data/parent/belief_hiddenstate_distribution/epoch_{epoch}.csv", belief_hiddenstate_distribution, fmt="%.5f")
    for x in range(0, hidden_state_range):
        belief_hiddenstate_distribution[a, x] = belief_conditional_hiddenstate_distribution[a, x, y_signal]
    return belief_hiddenstate_distribution


# xを更新
def xUpdate(belief_conditional_hiddenstate_distribution, action, y_signal):
    x = np.argmax(belief_conditional_hiddenstate_distribution[action][:][y_signal])
    relation = x % 5
    emotion = (x - relation) // 5
    return np.array([emotion, relation])

# p~(y)を更新
def preference_distribution_Update(preference_distribution, hidden_state):
    preference_distribution = np.array([1/15, 2/15, 3/15, 4/15, 5/15])
    return preference_distribution



def parent_inference(parent, epoch):
    # FE,ev,ps初期化
    parent.epistemic_value = np.zeros(action_range)
    parent.predicted_surprised = np.zeros(action_range)
    parent.F_expected = np.zeros(action_range)

    parent.belief_sensory_distribution = belief_sensory_distribution_make(parent.likelihood_distribution, parent.belief_hiddenstate_distribution, parent.belief_sensory_distribution, epoch) #q(y|x,a)
    parent.belief_conditional_hiddenstate_distribution = belief_conditional_hiddenstate_distribution_make(parent.likelihood_distribution, parent.belief_hiddenstate_distribution, parent.belief_sensory_distribution, parent.belief_conditional_hiddenstate_distribution, epoch) #q(x|y,a)

    # 隠れ状態xの更新
    parent.hidden_state = xUpdate(parent.belief_conditional_hiddenstate_distribution, parent.action, parent.sensory)
    # 選好分布の更新
    parent.preference_distribution = preference_distribution_Update(parent.preference_distribution, parent.hidden_state)

    # FE計算
    parent.epistemic_value = epistemic_value_calculate(parent.belief_hiddenstate_distribution, parent.belief_sensory_distribution, parent.belief_conditional_hiddenstate_distribution, parent.epistemic_value, epoch)
    parent.predicted_surprised = predicted_surprised_calculate(parent.belief_sensory_distribution, parent.preference_distribution, parent.predicted_surprised, epoch)
    parent.F_expected = F_expected_calculate(parent.epistemic_value, parent.predicted_surprised, parent.F_expected, epoch)
    parent.action = np.argmin(parent.F_expected)
    # q(x)の更新
    parent.belief_hiddenstate_distribution = qxUpdate(parent.belief_hiddenstate_distribution, parent.belief_conditional_hiddenstate_distribution, parent.sensory, parent.action, epoch)

    return parent