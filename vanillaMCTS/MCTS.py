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
    return self.state, reward, self.done

class Node:
  def __init__(self, state, parent=None, parent_action=None):
    self.state  = state
    self.parent = parent
    self.parent_action = parent_action
    self.wins = 0

    self.visits = 0
    self.children = []
    self.results  = {1:0, -1:0} 
    self._untried_actions = self.untried_actions()
    return

  # number of untried actions from given state
  def untried_actions(self):
    self._untried_actions = [i for i in range(9) if self.state[i] == 0]
    return self._untried_actions

  def UCTSelectChild(self):
    """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
        lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
        exploration versus exploitation.
    """
    s = sorted(self.children, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
    return s

  def AddChild(self, m, s):
    """ Remove m from untriedMoves and add a new child node for this move.
        Return the added child node
    """
    n = Node(state =self.state, parent = self,  parent_action=m)
    self._untried_actions.remove(m)
    self.children.append(n)
    return n
  def Update(self, result):
    """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
    """
    self.visits += 1
    self.wins += result

def UCT(rootstate, itermax, verbose = False):
  rootnode = Node(state = rootstate)
  _env = TicTacToe(rootstate)
  for i in range(itermax):
    node = rootnode
    state = rootstate.copy()

    # Select
    while node._untried_actions == [] and node.children != []: # node is fully expanded and non-terminal
      #print( "select" )
      node = node.UCTSelectChild()
    
    # Expand
    _env = TicTacToe( state )
    if node._untried_actions != []: # if we can expand (i.e. state/node is non-terminal)
      #print( "expand" )
      m = random.choice(node._untried_actions) 
      state, _,_, = _env.step(m)
      node = node.AddChild(m,state) # add child and descend tree

    # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
    #print( "rollout" )
    done = False
    reward = 0
    _env = TicTacToe( state )
    while not done: # while game not done contiune playing
      #_env.render()
      m = np.random.randint(0,9)
      obs, rew, done = _env.step(m)
      reward += rew

    #print( "Back Propagate" )
    # Backpropagate
    while node != None: # backpropagate from the expanded node and work back to the root node
      node.Update( reward ) # state is terminal. Update node with result from POV of node.playerJustMoved
      node = node.parent

  return np.random.randint(0,9)

env = TicTacToe()
state = env.reset()
done = False
while not done:
  env.render()
  m = UCT(rootstate = state, itermax = 100, verbose = False)
  print( "Best Move: " + str(m) + "\n" )
  obs, reward, done = env.step(m)
env.render()

