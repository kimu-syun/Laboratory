# 子供の能動推論(P_talkに対応)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import log
import random
import pandas as pd



# x,y,aの範囲
emotion_range = 5
relation_range = 5
sensory_range = 5
action_range = 5
hidden_state_range = emotion_range * relation_range


# xの変換(p(y|x,a))
def trans_gx(x):
    out = 0
    if x == 0:
        out = -0.6
    elif x == 1:
        out = -0.3
    elif x == 2:
        out = 0
    elif x == 3:
        out = 0.3
    elif x == 4:
        out = 0.6
    return out



# # 信念分布q(x|a)
# def Qxa_make(Qxa):
#     for a in range(0, 5):
#         sum = 0
#         for x in range(0, 5):
#             Qxa[x, a] = round(stats.norm.pdf(x=x, loc=a, scale=0.8), 4)
#             sum += Qxa[x, a]
#         # 正規化
#         for x in range(0, 5):
#             Qxa[x, a] = round(Qxa[x, a] / float(sum), 4)
#             if Qxa[x, a] == 0:
#                 Qxa[x, a] = 0.001
#     return Qxa


# 信念分布q(x|a)
# 最初を全部2中心の正規分布にする
def Qxa_make(Qxa):
    for a in range(0, 5):
        sum = 0
        for x in range(0, 5):
            Qxa[x, a] = round(stats.norm.pdf(x=x, loc=2, scale=0.4), 4)
            sum += Qxa[x, a]
        # 正規化
        for x in range(0, 5):
            Qxa[x, a] = round(Qxa[x, a] / float(sum), 4)
            if Qxa[x, a] == 0:
                Qxa[x, a] = 0.001
    return Qxa

# 尤度分布p(y|x,a)
def Pyxa_make(Pyxa):
    for y in range(0, 5):
        sum = 0
        for x in range(0, 5):
            for a in range(0, 5):
                Pyxa[y, x, a] = round(stats.norm.pdf(x=y, loc=a+trans_gx(x), scale=1.0), 4)
                sum += Pyxa[y, x, a]
        # 正規化
        for x in range(0, 5):
            for a in range(0, 5):
                Pyxa[y, x, a] = round(Pyxa[y, x, a] / float(sum), 4)
                if Pyxa[y, x, a] == 0:
                    Pyxa[y, x, a] = 0.001
    return Pyxa

# 選好分布(p~(y))
def Py_make(Py):
    Py[0] = np.array([0.001, 0.001, 1/3, 1/3, 1/3])
    Py[1] = np.array([0.001, 1/4, 1/4, 1/4, 1/4])
    Py[2] = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    Py[3] = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    Py[4] = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    return Py

# q(y|a)
def Qya_make(Qya, Pyxa, Qxa):
    for y in range(0, 5):
        for a in range(0, 5):
            for x in range(0, 5):
                Qya[y, a] += Pyxa[y, x, a] * Qxa[x, a] # q(y|a) = Σ_x{p(y|x)*q(x)}
            if Qya[y, a] == 0:
                Qya[y, a] = 0.001
    return Qya


# q(x|y,a)
def Qxya_make(Qxya, Pyxa, Qxa, Qya):
    for x in range(0, 5):
        for y in range(0, 5):
            for a in range(0, 5):
                Qxya[x, y, a] = round((Pyxa[y, x, a] * Qxa[x, a]) / float(Qya[y, a]), 4) # q(x|y,a) = {p(y|x,a)q(x|a)}/q(y|a)
                if Qxya[x,  y, a] == 0:
                    Qxya[x, y, a] = 0.001
    return Qxya


# epistemic value(エピスティミック価値)信念の更新度合い
def epistemic_value_calculate(Qya, Qxya, Qxa, epistemic_value):
    for a in range(0, 5):
        for y in range(0, 5):
            for x in range(0, 5):
                epistemic_value[a] = Qya[y, a] * Qxya[x, y, a] * log(Qxya[x, y, a] / float(Qxa[x, a])) # Σ_y{q(y|a)Σ_x{q(x|y,a)log(q(x|y,a)/q(x|a))}}
    return epistemic_value


# predicted surprised(予測サプライズ)
def predicted_surprised_calculate(Qya, Py, s, predicted_surprised):
    for a in range(0, 5):
        for y in range(0, 5):
            predicted_surprised[a] -= Qya[y][a] * log(Py[s][y])
    return predicted_surprised


# 期待自由エネルギーを計算、保存
def F_expected_calculate(epistemic_value, predicted_surprised, F_expected):
    for a in range(0, 5):
        F_expected[a] = -epistemic_value[a] + predicted_surprised[a]
    return F_expected


# q(x)を更新 q(x|y,a)->q(x)
def qxUpdate(Qx, Qxya, y, a):
    for x in range(0, 5):
        Qx[x, a] = Qxya[x, y, a]
    return Qx




def child_inference(child):
    # FE,ev,ps初期化
    child.epistemic_value = np.zeros(action_range)
    child.predicted_surprised = np.zeros(action_range)
    child.F_expected = np.zeros(action_range)

    child.Qya = Qya_make(child.Qya, child.Pyxa, child.Qxa) #q(y|a)
    child.Qxya = Qxya_make(child.Qxya, child.Pyxa, child.Qxa, child.Qya) #q(x|y,a)

    # 隠れ状態xの更新
    # child.hidden_state = xUpdate(child.Qxya, child.action, child.sensory)
    # 選好分布の更新

    # FE計算
    child.epistemic_value = epistemic_value_calculate(child.Qya, child.Qxya, child.Qxa, child.epistemic_value)
    child.predicted_surprised = predicted_surprised_calculate(child.Qya, child.Py, child.y, child.predicted_surprised)
    child.F_expected = F_expected_calculate(child.epistemic_value, child.predicted_surprised, child.F_expected)
    child.a = np.argmin(child.F_expected)
    # q(x)の更新
    child.Qxa = qxUpdate(child.Qxa, child.Qxya, child.y, child.a)

    return child