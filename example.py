from main import myTrader
import time
import pandas as pd
import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from tqdm import trange

if __name__ == "__main__":

    data=pd.read_csv('BTC.csv')
    data.drop(data.columns[0],1, inplace=True)
    # data=data.loc[:100]
    env=myTrader(data, 1000)
    # env.reset()
    # for _ in trange(100):
    #     obs, reward, done, _ = env.step(env.action_space.sample())
    #     if done==True:
    #         break
    #     env.render()

  
    def make_env(env: gym.Env, rank: int, seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """
        def _init() -> gym.Env:
            env.seed(seed + rank)
            return env
        set_random_seed(seed)
        return _init

    params = { "learning_rate": 1e-5}
    vec_env = SubprocVecEnv([make_env(env, i) for i in range(4)])

    agent = A2C('MlpPolicy', vec_env, verbose=0)
    # agent = A2C(MlpPolicy, env, n_steps=1000, **params)

    agent.learn(total_timesteps=1000)
