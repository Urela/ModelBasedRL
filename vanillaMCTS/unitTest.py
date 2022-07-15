import numpy as np
import random
from math import sqrt, log

class TicTacToe():
  def __init__(self, state=None):
    self.reset()
    if state is not None:
      self.state = state
      self.done = False
    pass

  def reset(self):
    self.done = False
    self.state = [0] *11
    self.state[-1] = 1   # start with player 1
    return self.state

  def render(self):
    print("turn %d" % self.state[-1])
    print(np.array(self.state[0:9]).reshape(3,3))

  # [0->9 board,  10 win or lose, 11  player]
  class observation_space():
    shape = (11,)
  class action_space():
    n = 9

  def value(self):
    for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
      if abs(self.state[x] + self.state[y] + self.state[z]) == 3:
        if self.state[x] == self.state[-1]:
          return 1  # win
        else:
          return -1 # loss

    if 0 not in self.state[0:9]: 
      return 0.5 # draw
    return 0  # game not over keep playing

  def dynamics(self, state, action):
    reward = 0
    # move in only empty spots
    if self.state[action] == 0:
      self.state[action] = self.state[-1] # make play
      reward = self.value()
    else: reward = -10
    return reward, state

  def step(self, action):
    #if reward = 0.5: done = True
    reward, self.state = self.dynamics(self.state, action)
    self.state[-1] = - self.state[-1]   # change players
    # game is over 
    if reward in [-1,0.5,1]:
      self.done = True
      self.state[-2] = 1
    return self.state, reward, self.done, None

win  = [4,0,3,1,6,2]     
draw = [0,1,2,6,3,5,7,4,8] 
for policy in [win, draw]:
  test = policy.copy()
  env = TicTacToe()
  done = False
  while not done:
    env.render()
    m = policy[0]
    policy.remove(m)
    state, reward, done, _ = env.step(m)
    #print(state)
  env.render()
  print("Test ",test, " Passed")

#a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]                     
#b = ([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1], 0, False, None)  
#c = ([-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 0, False, None)  
#d = ([-1, 0, 0, 1, 1, 0, 0, 0, 0, 0, -1], 0, False, None) 
#e = ([-1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 1], 0, False, None) 
#f = ([-1, -1, 0, 1, 1, 0, 1, 0, 0, 0, -1], 0, False, None)
#g = ([-1, -1, -1, 1, 1, 0, 1, 0, 0, 1, 1], 1, True, None) 
#
#print("Test", 'Passed' if a == env.reset() else "Failed", a)
#print("Test", 'Passed' if b == env.step(4) else "Failed", b)
#print("Test", 'Passed' if c == env.step(0) else "Failed", c)
#print("Test", 'Passed' if d == env.step(3) else "Failed", d)
#print("Test", 'Passed' if e == env.step(1) else "Failed", e)
#print("Test", 'Passed' if f == env.step(6) else "Failed", f)
#print("Test", 'Passed' if g == env.step(2) else "Failed", g)
#
