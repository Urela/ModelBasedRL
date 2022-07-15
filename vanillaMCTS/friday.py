import random 
import numpy as np
import ast
import math



def clone(obs):
  return obs.copy()

def hasMovesLeft(obs):
  return 0 in obs[0:9]

def getNextMoves(obs):
  a = [i for i in range(9) if obs[i] == 0]
  random.shuffle( a )
  return a 

def hasWon(obs): 
  for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
    if abs(obs[x] + obs[y] + obs[z]) == 3:
      if obs[x] == obs[-1]: # win
        obs[-2] = 1 #mark as done
        return obs[-1] # return who won
  
  if 0 not in state[0:9]: 
    obs[-2] = 1 #mark as done
    return 0.5 # draw
  return  0 # game not over yet

def render(obs):
  print("turn %d" % obs[-1])
  print(np.array(obs[0:9]).reshape(3,3))

def getBestNextMove(obs): 
  evaluations = {}
  epochs = 3
  for epi in range(epochs):
    _state = clone(obs)
    nextMoves = getNextMoves(obs)

    ## Rollout
    score = 9 # max score
    simulationMoves = []
    while nextMoves != []:
      action = nextMoves.pop(0)       # get acion
      _state[action] = _state[-1]    # make play
      _state[-1] *= -1               # change players
      #print(_state, action)
      simulationMoves.append(_state.copy())

      if hasWon(_state) != 0:
        score *= state[-1]
        break
      
      # Fake backprop
      score -= 1
      #nextMoves = getNextMoves(obs)   ## ???

    #for p in simulationMoves: print( p)
    if simulationMoves == []: 
      _state[-2] = 1
      return _state

    firstMove = simulationMoves[0]
    #print( firstMove )
    firstMoveKey = repr(firstMove)
    if firstMoveKey in evaluations:
      evaluations[firstMoveKey] += score
    else:
      evaluations[firstMoveKey] = score

  bestMove = obs
  highestScore = 0
  firstRound = True

  for move, score in evaluations.items():
    if firstRound or score > highestScore:
      highestScore = score
      bestMove = ast.literal_eval(move)
      firstRound = False
  return bestMove


state = [0]*11
state[-1] = 1  # starting player

while( state[-2] == 0 ):
  render(state)
  nstate = getBestNextMove(state)
  state = nstate.copy()

#print( hasWon(state, -1) )
#nstate = clone(state)
#print(hasMovesLeft(nstate))
