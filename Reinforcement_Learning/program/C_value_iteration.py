### -*-coding:utf-8-*-
# 方策最適化(子供)
import numpy as np
import matplotlib.pyplot as plt
from program import C_grid_world


def child_DP(state, sensor):
    # agentの生成
    agent = C_grid_world.Agent(state)

    num_row = 5
    num_col = 5

    # Vの初期化
    V = np.zeros((num_row, num_col))
    # piの初期化
    pi = np.random.randint(0,len(agent.ACTIONS),(num_row,num_col)) # 確定的な方策



    N = 1000
    V_trend = np.zeros((N, num_row, num_col))
    pi_trend = np.zeros((N, num_row, num_col))

    entire_count = 0
    policy_stable = False

    while(policy_stable == False):
        while(True):
            # 方策評価
            # print('entire count %d: ' % entire_count)
            count = 0
            delta = 0
            for i in range(num_row):
                for j in range(num_col):
                    tmp = np.zeros(len(agent.ACTIONS))
                    v = V[i,j]
                    for index,action in enumerate(agent.ACTIONS):
                    #print("delta %f" % delta)
                        agent.set_pos([i,j])
                        s = agent.get_pos()
                        agent.move(action)
                        s_dash = agent.get_pos()
                        tmp[index] =  (agent.reward(s,s_dash,action,sensor) + agent.GAMMA * V[s_dash[0], s_dash[1]])
                    V[i,j] = max(tmp)
                    delta = max(delta, abs(v - V[i,j]))
            count += 1
            if delta < 1.E-5:
                break

        V_trend[entire_count, :,:] = V

        # 方策改善
        b = pi.copy()
        for i in range(num_row):
            for j in range(num_col):
                tmp = np.zeros(len(agent.ACTIONS))
                for index, action in enumerate(agent.ACTIONS):
                    agent.set_pos([i,j])
                    s = agent.get_pos()
                    agent.move(action)
                    s_dash = agent.get_pos()
                    tmp[index] =  (agent.reward(s,s_dash,action,sensor) + agent.GAMMA * V[s_dash[0], s_dash[1]])
                pi[i,j] = np.argmax(tmp)

        pi_trend[:count,:,:] = pi
        if(np.all(b==pi)):
            policy_stable = True
            break

        entire_count += 1


    action = pi[state[0], state[1]]
    agent.set_pos([state[0], state[1]])
    s = agent.get_pos()
    agent.move(agent.ACTIONS[action])
    next_state = agent.get_pos()

    return next_state, pi[state[0], state[1]]

    ## 結果をグラフィカルに表示
    #方策を矢印で表示
    # C_grid_world.pi_arrow_plot(pi)
    #状態価値関数を表示
    # C_grid_world.V_value_plot(V)