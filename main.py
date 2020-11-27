import gym
from gym import spaces
import numpy as np
import pandas as pd
from enum import Enum
from itertools import product
import matplotlib.pyplot as plt 
import time
from tqdm import trange, tqdm
from wallet import Wallet
from rewards import RiskAdjustedReturns

class TradingEnvAction(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

class myTrader(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, data, balance, commission=0.01, **kwargs):
    #TODO saskirot jau sakuma datus liste
    super(myTrader, self).__init__()
    self.data=data
    self.plt=None
    self.ax=None
    self.risk=RiskAdjustedReturns(window_size=5)
    self.base_precision: int = kwargs.get('base_precision', 2)
    self.asset_precision: int = kwargs.get('asset_precision', 8)
    self.window_size=kwargs.get('window_size', 1)
    self.wallet=Wallet(data, usd=balance, commission=commission)
    self.minimum_balance: int = kwargs.get('minimum_balance', 100)
    self.current_step=0
    self.history={'episodes':[], 'stocks':[], 'worth':[], 'action':[], 'buy':{'x':[], 'y':[]}, 'sell':{'x':[], 'y':[]}}
    self.n_discrete_actions: int = kwargs.get('n_discrete_actions', 24)
    amount=1.0/(self.n_discrete_actions-(self.n_discrete_actions*0.5))
    self.action=list(product(range(2), np.arange(amount, 1, amount)))
    self.action.append((0, 1.0))
    self.action.append((1, 1.0))
    self.action.append((2, 0))
    self.action_space = spaces.Discrete(len(self.action))
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.data.values[self.current_step].shape)

  def step(self, action):
    # Execute one time step within the environment
    done=False
    if TradingEnvAction.BUY==TradingEnvAction(self.action[action][0]) and self.wallet.buy(self.action[action][1]):
        self.history['buy']['x'].append([self.current_step])
        self.history['buy']['y'].append(self.data['close'][self.current_step])   

    elif TradingEnvAction.SELL==TradingEnvAction(self.action[action][0]) and self.wallet.sell(self.action[action][1]):
        self.history['sell']['x'].append([self.current_step])
        self.history['sell']['y'].append( self.data['close'][self.current_step]) 

    self.history['action'].append([TradingEnvAction(self.action[action][0]), self.action[action][1]])
    self.history['episodes'].append(self.current_step)
    self.history['stocks'].append(self.data['close'][self.current_step])
    self.history['worth'].append(self.wallet.total())
        
    reward=self.risk.get_reward(self.history['worth'])
    if self.wallet.total()<=self.minimum_balance or len(self.data)-1==self.current_step:
        done=True 
    obs=self.data.values[self.current_step]
    self.current_step += 1
    self.wallet.step()
    return obs, reward, done, {}

  def _observation(self):
    if (self.current_step-self.window_size)<0:
      obs=np.zeros((self.observation_space.shape))
      obs[-(self.current_step+1):]=self.data.values[0:(self.current_step+1)]
    else:
      obs=self.data.values[((self.current_step+1)-self.window_size):self.current_step+1]
    return obs
  def reset(self):
    self.current_step=0
    self.wallet.reset()
    self.history={'episodes':[], 'stocks':[], 'worth':[], 'action':[], 'buy':{'x':[], 'y':[]}, 'sell':{'x':[], 'y':[]}}
    obs=self.data.values[self.current_step]
    self.history['episodes'].append(self.current_step)
    self.history['worth'].append(self.wallet.total())
    self.history['action'].append([TradingEnvAction(2), 0])
    self.history['stocks'].append(self.data['close'][self.current_step])
    self.current_step += 1
    self.wallet.step()

    return obs

  def render(self, mode='human'):
    # Render the environment to the screen
    if mode == 'system':
      tqdm.write('Episode:{0} worth:{1} Action:{2} amount:{3}'.format(self.history['episodes'][-1], self.history['worth'][-1], self.history['action'][-1][0], self.history['action'][-1][1]))
    elif mode == 'human':
      if not self.plt:
        plt.ion()
        fig, ax=plt.subplots(2)
        self.ax=ax
        self.plt=plt
      self.ax[0].plot(self.history['stocks'], color='red')
      if self.history['buy']:
          self.ax[0].scatter(self.history['buy']['x'], self.history['buy']['y'], label='skitscat', color='green', s=100, marker="^")
      if self.history['sell']:
          self.ax[0].scatter(self.history['sell']['x'], self.history['sell']['y'], label='skitscat', color='red', s=100, marker="v")
      self.ax[1].plot(self.history['worth'], color='grey')
      self.plt.draw()
      self.plt.pause(0.01)



