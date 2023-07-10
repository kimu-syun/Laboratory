# 能動的推論を使った会話モデル
import numpy as np
from math import log
import program
import matplotlib.pyplot as plt

# agentの情報保存
class agent:

    def __init__(self):
        # x,y,a
        self.x = 0
        self.y = 0
        self.a = 0
        # FE求めるのに必要なやつ
        self.epistemic_value = np.zeros(a_range)
        self.predicted_surprised = np.zeros(a_range)
        self.F_expected = np.zeros(a_range)
        # 分布の設定
        self.Pyxa = np.zeros((y_range, x_range, a_range)) # p(y|x,a)
        self.Qxa = np.zeros((x_range, a_range)) # q(x|a)
        self.Py = np.zeros((5, y_range)) # p~(y)
        self.Qya = np.zeros((y_range, a_range)) # q(y|a)
        self.Qxya = np.zeros((x_range, y_range, a_range)) # q(x|y,a)

        # 記録
        self.F_save = np.zeros((5, epoch))
        self.PS_save = np.zeros((5, epoch))
        self.EV_save = np.zeros((5, epoch))




# x,y,aの範囲
emotion_range = 5
relation_range = 5
y_range = 5
a_range = 5
x_range = 5
hidden_state_range = emotion_range * relation_range

# active inference
epoch = 10


# 真の値
child = agent()
# parent = agent()


# 子供の分布の作成(p(y|x,a), q(x|a), p~(y))
child.Pyxa = program.C_talk.Pyxa_make(child.Pyxa)
child.Qxa = program.C_talk.Qxa_make(child.Qxa)
child.Py = program.C_talk.Py_make(child.Py)


# 親の関数
parent = program.P_talk.agent_function()
P_x = 2
P_a = 0
P_aini = 2



# parent.action = 1 #親の初期行動
P_a = P_aini

for i in range(0, epoch):
    print(f"{i+1}回目")

    # 親の行動⇒子の感覚
    child.y = P_a

    # 子の推論
    print(f"子供の感覚信号 : {child.y}")
    child = program.C_talk.child_inference(child)

    for j in range(0, 5):
        child.F_save[j, i] = child.F_expected[j]
        child.PS_save[j, i] = child.predicted_surprised[j]
        child.EV_save[j, i] = child.epistemic_value[j]

    print(f"子供の行動{child.a}, FE{child.F_expected}")

    P_y = child.a
    # 親の関数
    P_x, P_a = parent.fun(P_x, P_y)
    print(f"親 x={P_x}, a={P_a}")
    print()


# F出力
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f"Active Inference - parent initial action = {P_aini}")

ax1 = fig.add_subplot(2, 2, 1)

ax1.set_xlabel("time steps")
ax1.set_ylabel("期待自由エネルギーF")
ax1.grid()

ax1.plot(child.F_save[0], label = "action0")
ax1.plot(child.F_save[1], label = "action1")
ax1.plot(child.F_save[2], label = "action2")
ax1.plot(child.F_save[3], label = "action3")
ax1.plot(child.F_save[4], label = "action4")


ax2 = fig.add_subplot(2, 2, 2)
ax2.set_xlabel("time steps")
ax2.set_ylabel("epistimec value")
ax2.grid()
ax2.plot(child.EV_save[0], label = "action0")
ax2.plot(child.EV_save[1], label = "action1")
ax2.plot(child.EV_save[2], label = "action2")
ax2.plot(child.EV_save[3], label = "action3")
ax2.plot(child.EV_save[4], label = "action4")


ax3 = fig.add_subplot(2, 2, 3)
ax3.set_xlabel("time steps")
ax3.set_ylabel("predicted surprised")
ax3.grid()
ax3.plot(child.PS_save[0], label = "action0")
ax3.plot(child.PS_save[1], label = "action1")
ax3.plot(child.PS_save[2], label = "action2")
ax3.plot(child.PS_save[3], label = "action3")
ax3.plot(child.PS_save[4], label = "action4")


ax1.legend()
ax2.legend()
ax3.legend()
# ax1.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# ax2.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# ax3.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# plt.legend(["action0", "action1", "action2", "action3", "action4"])
plt.show()