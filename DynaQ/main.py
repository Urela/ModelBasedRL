import gym
import random
import collections
import numpy as np
from models import *

"""
There is a line where we chose how much of real expirance vs simulated experiance is 
stored in the replay buffer

  int a = 1
  if np.random.random() < a:
    agent.memory.append((obs, action, _reward.item(), _obs.detach().numpy(), done))
  else:
    agent.memory.append((obs, action, reward, next_obs, done))

  a = 1 only simulated (generated from model)
  a = (0,1) paritally real and simulated
  a = 0 only real  ( from OpenAI Gym

"""

# Global variables
env=gym.make('CartPole-v1')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

# Models
dynamicsNet = DynamicsModel(env.observation_space, env.action_space ).to('cpu')
rewardsNet  = RewardsModel( env.observation_space, env.action_space ).to('cpu')
terminalNet = RewardsModel( env.observation_space, env.action_space ).to('cpu')
class DQN:
  def __init__(self, in_space, out_space):
    self.lr = 1e-3
    self.gamma   = 0.99
    self.epsilon = 1.0
    self.eps_min = 0.05
    self.eps_dec = 5e-4
    self.action_size = out_space.n
    self.memory  = collections.deque(maxlen=100000)
    self.policy = Policy( in_space, out_space, self.lr).to('cpu')
    self.target = Policy( in_space, out_space, self.lr).to('cpu') 

  def update_target(self):
    self.target.load_state_dict( self.policy.state_dict() )

  def update_epsilon(self):
    self.epsilon = max(self.eps_min, self.epsilon*self.eps_dec)


  def train(self, batch_size=64):
    if len(self.memory) >= batch_size:
      for i in range(10):
        batch = random.sample(self.memory, batch_size)
        states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
        actions = torch.tensor([[x[1]] for x in batch])
        rewards = torch.tensor([[x[2]] for x in batch]).float()
        nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
        dones   = torch.tensor([x[4] for x in batch])

        q_pred = self.policy(states).gather(1, actions)
        q_targ = self.target(nstates).max(1)[0].unsqueeze(1)
        q_targ[dones] = 0.0  # set all terminal states' value to zero
        q_targ = rewards + self.gamma * q_targ 

        loss = F.smooth_l1_loss(q_pred, q_targ).to('cpu')
        self.policy.backprop(loss)
      return loss
    else:
      return torch.tensor([0])

agent = DQN( env.observation_space, env.action_space )

Ldynamics= []
Lrewards = []
Lterminal= []
Lagent   = []

scores = []
for episode in range( 500 ):
  score, steps = 0, 0
  obs, done = env.reset(), False
  while not done:

    # epsilon greedy stratergy
    if np.random.random() > agent.epsilon:
      _action = agent.policy(torch.tensor(obs, dtype=torch.float))
      action = _action.argmax().item()
    else:
      action = env.action_space.sample()
      _action = torch.tensor([0,0]); _action[action]=1

    next_obs, reward, done, info = env.step(action)

    # predict the world 
    _obs    = dynamicsNet( torch.tensor(obs).float(), _action).to('cpu')
    _reward = rewardsNet(  torch.tensor(obs).float(), _action).to('cpu')
    _done   = terminalNet( torch.tensor(obs).float(), _action).to('cpu')

    loss = F.smooth_l1_loss(_obs, torch.tensor(obs))
    ld = dynamicsNet.backprop(loss)

    # train the various models
    loss = F.smooth_l1_loss(_reward, torch.tensor([reward]))
    lr = rewardsNet.backprop(loss)

    loss = F.smooth_l1_loss(_done, torch.tensor([int(done)]))
    lt = terminalNet.backprop(loss)

    if np.random.random() < 1:
      #_done = int(abs(_done)/0.5)
      #agent.memory.append((obs, action, _reward.item(), _obs.detach().numpy(), _done))
      agent.memory.append((obs, action, _reward.item(), _obs.detach().numpy(), done))
    else:
      print('true')
      agent.memory.append((obs, action, reward, next_obs, done))

    la = agent.train()

    # Update episodic varibles
    steps += 1
    score += reward
    obs = next_obs

    Ldynamics.append(ld.item())
    Lrewards.append(lr.item())
    Lterminal.append(lt.item())
    Lagent.append(la.item())


  agent.update_epsilon() # update epsilon value after each episode
  agent.update_target()  # update target network after each episode

  # Record episodic states
  scores.append(score)
  avg_score = np.mean(scores[-10:]) # moving average of last 100 episodes
  print(f"Episode {episode}, Return: {scores[-1]}, Avg return: {avg_score}")

env.close()

from bokeh.plotting import figure, show
# create a new plot with a title and axis labels
p = figure(title="dynamics & reward model losses", x_axis_label="Episode", y_axis_label="loss")
# add a line renderer with legend and line thickness
p.line(np.arange(len(Lterminal)), Lterminal, legend_label="Terminal state model loss", line_color="orange", line_width=2)
p.line(np.arange(len(Ldynamics)), Ldynamics,  legend_label="Dynamics model loss", line_color="red", line_width=2)
p.line(np.arange(len(Lagent)), Lagent, legend_label="Agents Loss", line_color="green", line_width=2)
p.line(np.arange(len(Lrewards)), Lrewards, legend_label="Rewards model loss", line_color="blue", line_width=2)
show(p) # show the results

#from bokeh.plotting import figure, show
## create a new plot with a title and axis labels
#p = figure(title="Scores", x_axis_label="Episode", y_axis_label="loss")
## add a line renderer with legend and line thickness
#p.line(np.arange(len(scores)), scores, legend_label="Score", line_color="blue", line_width=2)
#show(p) # show the results
