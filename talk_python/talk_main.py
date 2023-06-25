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


# 分布の作成(p(y|x,a), q(x|a), p~(y))
child.likelihood_distribution = program.talk_child.likelihood_distribution_make(child.likelihood_distribution)
child.belief_hiddenstate_distribution = program.talk_child.belief_hiddenstate_distribution_make(child.belief_hiddenstate_distribution)
child.preference_distribution = program.talk_child.preference_distribution_make(child.preference_distribution)


# active inference
count = 10

for i in range(0, count):
    print(f"{i+1}回目")
    # y_signal = int(input("感覚信号の入力 : "))
    child.sensory = random.randint(0, 4)
    print(f"感覚信号 : {child.sensory}")
    child = program.talk_child.child_inference(child, child.sensory)
    print(f"action{child.action}, FE{child.F_expected}")
