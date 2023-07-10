# 対話モデルをQ_learningで実装
# 参考(https://github.com/triwave33/reinforcement_learning)
import random
import numpy as np
import program
import matplotlib.pyplot as plt


time_steps = 12
# 初期値
child_state = [1]

parent = program.P_talk.agent_function()
P_x = 2
P_a = [0]
P_aini = 4
P_aini = [P_aini]

pi_save = np.zeros((5, time_steps))
V_save = np.zeros((5, time_steps))

P_a = P_aini

for t in range(0, time_steps):
    print(f"{t+1}回目")
    child_state = [P_a]
    child_state, child_action, pi, V = program.C_value_iteration.child_DP(child_state)

    for i in range(0, 5):
        pi_save[i, t] = pi[i]
        V_save[i, t] = V[i]


    print(f"子ども action-{child_action}-, child_state-{child_state}-")

    P_y = int(child_action)
    # 親の関数
    P_x, P_a = parent.fun(P_x, P_y)
    print(f"親 x:{P_x}, a:{P_a}")
    P_a = [P_a]

# F出力
fig = plt.figure(figsize=(8, 4))
fig.suptitle(f"Reinforcement Learning - parent initial action = {P_aini[0]}")

ax1 = fig.add_subplot(1, 2, 1)

ax1.set_xlabel("time steps")
ax1.set_ylabel("Policy π")
ax1.grid()

ax1.plot(pi_save[0], label = "action0")
ax1.plot(pi_save[1], label = "action1")
ax1.plot(pi_save[2], label = "action2")
ax1.plot(pi_save[3], label = "action3")
ax1.plot(pi_save[4], label = "action4")


ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel("time steps")
ax2.set_ylabel("Value V")
ax2.grid()
ax2.plot(V_save[0], label = "action0")
ax2.plot(V_save[1], label = "action1")
ax2.plot(V_save[2], label = "action2")
ax2.plot(V_save[3], label = "action3")
ax2.plot(V_save[4], label = "action4")




ax1.legend()
ax2.legend()
# ax1.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# ax2.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# ax3.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# plt.legend(["action0", "action1", "action2", "action3", "action4"])
plt.show()
