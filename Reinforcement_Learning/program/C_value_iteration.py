### -*-coding:utf-8-*-
# 方策最適化(子供)
import numpy as np
import matplotlib.pyplot as plt
from program import C_grid_world


def child_DP(state):
    # agentの生成
    agent = C_grid_world.Agent(state)

    num_row = 5
    num_col = 5

    # Vの初期化
    V = np.zeros((num_row))
    # piの初期化
    pi = np.random.randint(0,len(agent.ACTIONS),(num_row)) # 確定的な方策



    N = 1000
    V_trend = np.zeros((N, num_row))
    pi_trend = np.zeros((N, num_row))

    entire_count = 0
    policy_stable = False

    while(policy_stable == False):
        while(True):
            # 方策評価
            # print('entire count %d: ' % entire_count)
            count = 0
            delta = 0
            for i in range(num_row):
                tmp = np.zeros(len(agent.ACTIONS))
                v = V[i]
                for index,action in enumerate(agent.ACTIONS):
                #print("delta %f" % delta)
                    agent.set_pos([i])
                    s = agent.get_pos()
                    agent.move(action)
                    s_dash = agent.get_pos()
                    tmp[index] =  (agent.reward(s,s_dash,action) + agent.GAMMA * V[s_dash[0]])
                V[i] = max(tmp)
                delta = max(delta, abs(v - V[i]))
            count += 1
            if delta < 1.E-5:
                break

        V_trend[entire_count,:] = V

        # 方策改善
        b = pi.copy()
        for i in range(num_row):
            tmp = np.zeros(len(agent.ACTIONS))
            for index, action in enumerate(agent.ACTIONS):
                agent.set_pos([i])
                s = agent.get_pos()
                agent.move(action)
                s_dash = agent.get_pos()
                tmp[index] =  (agent.reward(s,s_dash,action) + agent.GAMMA * V[s_dash[0]])
            pi[i] = np.argmax(tmp)

        pi_trend[:count,:] = pi
        if(np.all(b==pi)):
            policy_stable = True
            break

        entire_count += 1


    action = pi[state[0]]
    agent.set_pos([state[0]])
    s = agent.get_pos()
    agent.move(agent.ACTIONS[action[0]])
    next_state = agent.get_pos()

    return next_state, pi[state[0]], pi, V

    ## 結果をグラフィカルに表示
    #方策を矢印で表示
    # C_grid_world.pi_arrow_plot(pi)
    #状態価値関数を表示
    # C_grid_world.V_value_plot(V)
