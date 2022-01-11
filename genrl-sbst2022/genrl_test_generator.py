import logging as log

from stable_baselines3 import PPO

from road_generation_env_continuous import RoadGenerationContinuousEnv
from road_generation_env_discrete import RoadGenerationDiscreteEnv
from road_generation_env_transform import RoadGenerationTransformationEnv

from code_pipeline.tests_generation import RoadTestFactory


class CarlTestGenerator:
    """
        Generates tests using a RL-based approach
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):
        log.info("Starting CaRL test generator")

        # Instantiate the environment
        # env = RoadGenerationContinuousEnv(test_executor, max_number_of_points=20)
        # env = RoadGenerationDiscreteEnv(test_executor, max_number_of_points=8)
        env = RoadGenerationTransformationEnv(self.executor, max_number_of_points=4)

        # Instantiate the agent
        model = PPO('MlpPolicy', env, verbose=1)

        # Start training the agent
        model.learn(total_timesteps=int(1e2))

        # If training is done and we still have time left, we generate new tests using the trained policy until the
        # given time budget is up.
        while True:
            obs = env.reset()
            while not done:
                action = model.predict(observation=obs)
                obs, reward, done, info = env.step(action)
