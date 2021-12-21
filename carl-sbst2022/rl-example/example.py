import gym
import os

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('LunarLander-v2')  # CartPole-v1

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1, device="cuda", batch_size=128)
cwd = os.getcwd()
print(cwd)
# Train the agent (if not already saved)
if not os.path.isfile("dqn_lunar.zip"):
    print("Training model")
    model.learn(total_timesteps=int(2e5), log_interval=100)
    model.save("dqn_lunar")
else:
    print("Skipped training. Model already exists")
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
