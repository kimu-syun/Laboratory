# 子供の能動推論
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
def belief_sensory_distribution_make(likelihood_distribution, belief_hiddenstate_distribution, belief_sensory_distribution):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                belief_sensory_distribution[a, y] = \
                    belief_sensory_distribution[a, y] + (likelihood_distribution[a, x, y] * belief_hiddenstate_distribution[a, x]) # q(y|a) = Σ_x{p(y|x)*q(x)}
    return belief_sensory_distribution


# q(x|y,a)
def belief_conditional_hiddenstate_distribution_make(likelihood_distribution, belief_hiddenstate_distribution, belief_sensory_distribution, belief_conditional_hiddenstate_distribution):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                belief_conditional_hiddenstate_distribution[a, x, y] = \
                    ((likelihood_distribution[a, x, y] * belief_hiddenstate_distribution[a, x]) / belief_sensory_distribution[a, y]) # q(x|y,a) = {p(y|x,a)q(x|a)}/q(y|a)
    return belief_conditional_hiddenstate_distribution


# epistemic value(エピスティミック価値)信念の更新度合い
def epistemic_value_calculate(belief_hiddenstate_distribution, belief_sensory_distribution, belief_conditional_hiddenstate_distribution, epistemic_value):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                epistemic_value[a] = belief_sensory_distribution[a, y] * belief_conditional_hiddenstate_distribution[a, x, y]\
                      * log((belief_conditional_hiddenstate_distribution[a, x, y]) / belief_hiddenstate_distribution[a, x]) # Σ_y{q(y|a)Σ_x{q(x|y,a)log(q(x|y,a)/q(x|a))}}
    return epistemic_value



# predicted surprised(予測サプライズ)
def predicted_surprised_calculate(belief_sensory_distribution, preference_distribution, predicted_surprised):
    for a in range(0, action_range):
        for y in range(0, sensory_range):
            for x in range(0, hidden_state_range):
                predicted_surprised[a] = predicted_surprised[a] - belief_sensory_distribution[a, y] * log(preference_distribution[y])
    return predicted_surprised



# 期待自由エネルギーを計算、保存
def F_expected_calculate(epistemic_value, predicted_surprised, F_expected):
    for a in range(0, action_range):
        F_expected[a] = -epistemic_value[a] + predicted_surprised[a]
    return F_expected


# q(x)を更新 q(x|y,a)->q(x)
def qxUpdate(belief_hiddenstate_distribution, belief_conditional_hiddenstate_distribution, y_signal, a):
    for x in range(0, hidden_state_range):
        belief_hiddenstate_distribution[a, x] = belief_conditional_hiddenstate_distribution[a, x, y_signal]
    return belief_hiddenstate_distribution


def child_inference(child, y_signal):
    child.belief_sensory_distribution = belief_sensory_distribution_make(child.likelihood_distribution, child.belief_hiddenstate_distribution, child.belief_sensory_distribution) #q(y|x,a)
    child.belief_conditional_hiddenstate_distribution = belief_conditional_hiddenstate_distribution_make(child.likelihood_distribution, child.belief_hiddenstate_distribution, child.belief_sensory_distribution, child.belief_conditional_hiddenstate_distribution) #q(x|y,a)

    # FE計算
    child.epistemic_value = epistemic_value_calculate(child.belief_hiddenstate_distribution, child.belief_sensory_distribution, child.belief_conditional_hiddenstate_distribution, child.epistemic_value)
    child.predicted_surprised = predicted_surprised_calculate(child.belief_sensory_distribution, child.preference_distribution, child.predicted_surprised)
    child.F_expected = F_expected_calculate(child.epistemic_value, child.predicted_surprised, child.F_expected)
    child.action = np.argmin(child.F_expected)
    # q(x)更新
    child.belief_hiddenstate_distribution = qxUpdate(child.belief_hiddenstate_distribution, child.belief_conditional_hiddenstate_distribution, y_signal, child.action)

    return child