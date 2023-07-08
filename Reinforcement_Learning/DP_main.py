# 対話モデルをQ_learningで実装
# 参考(https://github.com/triwave33/reinforcement_learning)
import random
import numpy as np
import program


time_steps = 12
# 初期値
child_state = [1,3]

parent_state = [1]
parent_action = 1


for t in range(0, time_steps):
    print(f"{t+1}回目")
    child_state, child_action = program.C_value_iteration.child_DP(child_state, parent_action)
    print(f"子ども action-{child_action}-, child_state-{child_state}-")

    parent_state = [child_action]
    parent_state, parent_action = program.P_value_iteration.parent_DP(parent_state, child_action)
    print(f"親 action-{parent_action}-, parent_state-{parent_state}-")
