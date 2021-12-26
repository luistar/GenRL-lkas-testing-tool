import logging

import gym
from stable_baselines3 import DQN

from road_generation_env import RoadGenerationEnv

logging.basicConfig(level=logging.DEBUG)

env = RoadGenerationEnv()

# Instantiate the agent
# model = DQN('MlpPolicy', env, verbose=1, device="cuda", batch_size=128)

# model.learn(total_timesteps=int(2e5), log_interval=100)

# Enjoy trained agent
obs = env.reset()
for i in range(10):
    action = env.action_space.sample()
    print(str(action))
    obs, rewards, dones, info = env.step(action)
    print(str(obs))
    #env.render()


print("foo")
