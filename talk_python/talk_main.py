# 能動的推論を使った会話モデル
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib
from math import log
import random
import pandas as pd

import program

# agentの情報保存
class agent:

    def __init__(self):
        # x,y,a
        self.hidden_state = np.array([0, 0])
        self.sensory = 0
        self.action = 0
        # FE求めるのに必要なやつ
        self.epistemic_value = np.zeros(action_range)
        self.predicted_surprised = np.zeros(action_range)
        self.F_expected = np.zeros(action_range)
        # 分布の設定
        self.likelihood_distribution = np.zeros((action_range, hidden_state_range, sensory_range)) # p(y|x,a)
        self.belief_hiddenstate_distribution = np.zeros((action_range, hidden_state_range)) # q(x|a)
        self.preference_distribution = np.zeros((sensory_range)) # p~(y)
        self.belief_sensory_distribution = np.zeros((action_range, sensory_range)) # q(y|a)
        self.belief_conditional_hiddenstate_distribution = np.zeros((action_range, hidden_state_range, sensory_range)) # q(x|y,a)




# x,y,aの範囲
emotion_range = 5
relation_range = 5
sensory_range = 5
action_range = 5
hidden_state_range = emotion_range * relation_range


# 真の値
child = agent()
parent = agent()


# 子供の分布の作成(p(y|x,a), q(x|a), p~(y))
child.likelihood_distribution = program.talk_child.likelihood_distribution_make(child.likelihood_distribution)
child.belief_hiddenstate_distribution = program.talk_child.belief_hiddenstate_distribution_make(child.belief_hiddenstate_distribution)
child.preference_distribution = program.talk_child.preference_distribution_make(child.preference_distribution)

# 親の分布の作成(p(y|x,a), q(x|a), p~(y))
parent.likelihood_distribution = program.talk_parent.likelihood_distribution_make(parent.likelihood_distribution)
parent.belief_hiddenstate_distribution = program.talk_parent.belief_hiddenstate_distribution_make(parent.belief_hiddenstate_distribution)
parent.preference_distribution = program.talk_parent.preference_distribution_make(parent.preference_distribution)


# active inference
epoch = 12
parent.action = 3 #親の初期行動

for i in range(0, epoch):
    print(f"{i+1}回目")

    # 親の行動⇒子の感覚
    a = parent.action
    s = a
    child.sensory = a

    # 子の推論
    print(f"子供の感覚信号 : {child.sensory}")
    child = program.talk_child.child_inference(child, i)
    print(f"子供  action{child.action}, FE{child.F_expected}")

    # 子の行動⇒親の感覚
    a = child.action
    s = a
    parent.sensory = s

    # 親の推論
    print(f"親の感覚信号 : {parent.sensory}")
    parent = program.talk_parent.parent_inference(parent, i)
    print(f"親    action{parent.action}, FE{child.F_expected}")
