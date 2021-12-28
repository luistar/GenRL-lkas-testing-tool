import math
import os
import random
import logging
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from code_pipeline.executors import MockExecutor
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.visualization import RoadTestVisualizer


class RoadGenerationEnv(gym.Env):
    """
        Description:
            The agent aims at generating tests for a lane-keeping system in a simulated environment.
            Each test is a sequence of points in a 200x200 map. The agent starts with an empty sequence of points.
            For any given state, the agent may choose to add/update or delete a point from the sequence.
        Source:
            The environment was created to compete in the SBST2022 Cyber-physical systems (CPS) testing competition.
        Reward:
             TODO define reward
        Starting State:
             The agent starts with an empty sequence of points.
        Episode Termination:
             The agent constructed a test on which the lane-keeping system fails
             Episode length is greater than TODO steps
        """

    ADD_UPDATE = 0
    REMOVE = 1

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self,  executor, max_steps=1000, grid_size=200, results_folder="results", max_number_of_points=5):

        self.step_counter = 0

        self.max_steps = max_steps
        self.grid_size = grid_size
        self.max_number_of_points = max_number_of_points
        self.executor = executor

        self.min_coordinate = 0.0
        self.max_coordinate = 1.0  # valid coordinates are (x, y) with x, y \in [min_coordinate, max_coordinate]

        self.max_speed = float('inf')
        self.failure_oob_threshold = 0.95
        self.min_oob_percentage = 0.0
        self.max_oob_percentage = 100.0

        # state is an empty sequence of points
        self.state = np.empty(self.max_number_of_points, dtype=object)
        for i in range(self.max_number_of_points):
            self.state[i] = (0, 0)  # (0,0) represents absence of information in the i-th cell

        self.low_coordinates = np.array([self.min_coordinate, self.min_coordinate], dtype=np.float32)
        self.high_coordinates = np.array([self.max_coordinate, self.max_coordinate], dtype=np.float32)
        self.low_observation = np.array([], dtype=np.float32)
        self.high_observation = np.array([], dtype=np.float32)

        self.viewer = None

        # self.action_space = spaces.Discrete(2) #  no bueno

        # looks like you can't mix discrete and box in tuples
        # self.action_space = spaces.Tuple(
        #     spaces.MultiDiscrete([2, self.number_of_points]),  # 2 actions (add/update and delete), one index position
        #     spaces.Box(self.low_observation, self.high_observation, dtype=np.float32) # new coordinates for the point
        # )

        # action space as a box
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, self.min_coordinate + 0.1, self.min_coordinate + 0.1]),
            high=np.array([1.0, float(self.max_number_of_points) - np.finfo(float).eps, self.max_coordinate - 0.1, self.max_coordinate - 0.1]),
            dtype=np.float16
        )

        # action space as a MultiDiscrete set (action type, position, x, y) # TODO make discrete env variant
        # self.action_space = spaces.MultiDiscrete([2, self.number_of_points, 1600, 1600])

        # create box observation space
        for i in range(self.max_number_of_points):
            self.low_observation = np.append(self.low_observation, [0.0, 0.0])
            self.high_observation = np.append(self.high_observation, [self.max_coordinate, self.max_coordinate])
        self.low_observation = np.append(self.low_observation, self.min_oob_percentage)
        self.high_observation = np.append(self.high_observation, self.max_oob_percentage)

        self.observation_space = spaces.Box(self.low_observation, self.high_observation, dtype=np.float16)

    def step(self, action):
        pass

    def reset(self, seed: Optional[int] = None):
        pass

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_road_points(self):
        pass

    def compute_reward(self, execution_data):
        reward = pow(execution_data[0].max_oob_percentage/10, 2)
        return reward
