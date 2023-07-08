import numpy as np
import random

class agent_function():

    alpha = 0.2
    x_dic ={0:-0.5, 1:-0.3, 2:0, 3:0.3, 4:0.5}
    y_dic ={0:-2, 1:-1, 2:0, 3:1, 4:2}

    def __init__(self):
        pass

    def trans(continu):
        # 連続値⇒離散値
        descrete = 0
        if -1.5 <= continu <= 0.5:
            descrete = 0
        elif 0.5 <= continu <= 1.5:
            descrete = 1
        elif 1.5 <= continu <= 2.5:
            descrete = 2
        elif 2.5 <= continu <= 3.5:
            descrete = 3
        elif 3.5 <= continu <= 5.5:
            descrete = 4

        return descrete

    def get_xy(self, x, y):
        #x,yを更新
        self.x = x
        self.y = y

    def action(self):
        # a_tを導出
        sigma = random.uniform(-0.5, 0.5)
        self.a = self.y + self.x_dic[self.x] + sigma
        self.a = self.trans(self.a)



    def update(self):
        # x_tをx_(t+1)に更新
        self.x = self.alpha*self.y_dic[self.y] + (1-self.alpha)*self.x

    def fun(self, x, y):
        # x_t, y_tを受け取って,x_(t+1),a_(t+1)を返す

        self.get_xy(x, y)
        self.action()
        self.update()

        return self.x, self.a
