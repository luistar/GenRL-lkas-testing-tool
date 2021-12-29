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

from road_generation_env import RoadGenerationEnv


class RoadGenerationContinuousEnv(RoadGenerationEnv):
    """
    Observation:
            Type: Box(2n+1) where n is self.number_of_points, the max number of points in the generated roads

            Num     Observation             Min                 Max
            0       x coord for 1st point   self.min_coord      self.max_coord
            1       y coord for 1st point   self.min_coord      self.max_coord
            2       x coord for 2nd point   self.min_coord      self.max_coord
            3       y coord for 2nd point   self.min_coord      self.max_coord
            ...
            2n-2    x coord for 2nd point   self.min_coord      self.max_coord
            2n-1    y coord for 2nd point   self.min_coord      self.max_coord
            n       Max %OOB                0.0                 1.0             # TODO fix

        Actions:
            Type: Box(4) ?
            Num     Action                  Min                 Max
            0       Action type             0                   1
            1       Position                0                   self.number_of_points
            2       New x coord             self.min_coord      self.max_coord
            3       New y coord             self.min_coord      self.max_coord
    """

    ADD_UPDATE = 0
    REMOVE = 1

    def __init__(self, executor, max_steps=1000, grid_size=200, results_folder="results", max_number_of_points=5,
                 max_reward=100, invalid_test_reward=-10):

        super().__init__(executor, max_steps, grid_size, results_folder, max_number_of_points, max_reward,
                         invalid_test_reward)

        self.min_coordinate = 0.0
        self.max_coordinate = 1.0

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

        # action space as a box
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, self.min_coordinate + 0.1, self.min_coordinate + 0.1]),
            high=np.array([1.0, float(self.max_number_of_points) - np.finfo(float).eps, self.max_coordinate - 0.1,
                           self.max_coordinate - 0.1]),
            dtype=np.float16
        )

        # create box observation space
        for i in range(self.max_number_of_points):
            self.low_observation = np.append(self.low_observation, [0.0, 0.0])
            self.high_observation = np.append(self.high_observation, [self.max_coordinate, self.max_coordinate])
        self.low_observation = np.append(self.low_observation, self.min_oob_percentage)
        self.high_observation = np.append(self.high_observation, self.max_oob_percentage)

        self.observation_space = spaces.Box(self.low_observation, self.high_observation, dtype=np.float16)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        self.step_counter = self.step_counter + 1  # increment step counter

        action_type = round(action[0])  # value in [0,1]
        position = math.floor(action[1])  # value in [0,self.number_of_points)
        x = action[2]  # coordinate in [self.min_coordinate,self.max_coordinate]
        y = action[3]  # coordinate in [self.min_coordinate,self.max_coordinate]

        logging.info(f"Processing action {str(action)}")

        if action_type == self.ADD_UPDATE and not self.check_coordinates_already_exist(x, y):
            logging.debug("Setting coordinates for point %d to (%.2f, %.2f)", position, x, y)
            self.state[position] = (x, y)
            reward, max_oob = self.compute_step()
        elif action_type == self.ADD_UPDATE and self.check_coordinates_already_exist(x, y):
            logging.debug("Skipping add of (%.2f, %.2f) in position %d. Coordinates already exist", x, y, position)
            reward = -10
            max_oob = 0.0
        elif action_type == self.REMOVE and self.check_some_coordinates_exist_at_position(position):
            logging.debug("Removing coordinates for point %d", position)
            self.state[position] = (0, 0)
            reward, max_oob = self.compute_step()
        elif action_type == self.REMOVE and not self.check_some_coordinates_exist_at_position(position):
            # disincentive deleting points where already there is no point
            logging.debug(f"Skipping delete at position {position}. No point there.")
            reward = self.invalid_test_reward
            max_oob = 0.0

        done = self.step_counter == self.max_steps

        # return observation, reward, done, info
        obs = [coordinate for tuple in self.state for coordinate in tuple]
        obs.append(max_oob)
        return np.array(obs, dtype=np.float16), reward, done, {}

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        # state is an empty sequence of points
        self.state = np.empty(self.max_number_of_points, dtype=object)
        for i in range(self.max_number_of_points):
            self.state[i] = (0, 0)  # (0,0) represents absence of information in the i-th cell
        # return observation
        obs = [coordinate for tuple in self.state for coordinate in tuple]
        obs.append(0.0)  # zero oob initially
        return np.array(obs, dtype=np.float16)

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
        road_points = []  # np.array([], dtype=object)
        for i in range(self.max_number_of_points):
            if self.state[i][0] != 0 and self.state[i][1] != 0:
                road_points.append(
                    (
                        self.state[i][0] * self.grid_size,
                        self.state[i][1] * self.grid_size
                    )
                )
        return road_points

    def compute_reward(self, execution_data):
        reward = pow(execution_data[0].max_oob_percentage / 10, 2)
        return reward

    def check_coordinates_already_exist(self, x, y):
        for i in range(self.max_number_of_points):
            if x == self.state[i][0] and y == self.state[i][1]:
                return True
        return False
