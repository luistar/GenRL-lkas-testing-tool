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
    Observation:
        Type: Box(2)
        Num    Observation          Min         Max
        0      Max %OOB             0.0         100.0

    Actions:
        Type: Discrete(2) ?
        Num    Action
        0      Add/Update points
        1      Remove point
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The agent starts with an empty sequence of points.
    Episode Termination:
         The agent constructed a test on which the lane-keeping system fails
         Episode length is greater than TODO steps
    """

    ADD_UPDATE = 0
    REMOVE = 1

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):

        self.grid_size = 200
        result_folder = "results"

        self.beamng_executor = MockExecutor(result_folder, time_budget=60, map_size=200,
                                            road_visualizer=RoadTestVisualizer(map_size=200))

        self.min_coordinate = 0.1
        self.max_coordinate = 0.9  # valid coordinates are (x, y) with x, y \in [min_coordinate, max_coordinate]
        self.number_of_points = 5
        self.max_speed = float('inf')

        self.failure_oob_threshold = 0.95

        self.min_oob_percentage = 0.0
        self.max_oob_percentage = 100.0

        # state is an empty sequence of points
        self.state = np.empty(self.number_of_points, dtype=object)
        for i in range(self.number_of_points):
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
            low=np.array([0.0, 0.0, self.min_coordinate, self.min_coordinate]),
            high=np.array([1.0, float(self.number_of_points), self.max_coordinate, self.max_coordinate]),
            dtype=np.float32
        )

        # action space as a MultiDiscrete set (action type, position, x, y) # TODO make discrete env variant
        # self.action_space = spaces.MultiDiscrete([2, self.number_of_points, 1600, 1600])

        # create box observation space
        for i in range(self.number_of_points):
            np.append(self.low_observation, 0.0)
            np.append(self.high_observation, self.max_coordinate)
        np.append(self.low_observation, self.min_oob_percentage)
        np.append(self.high_observation, self.max_oob_percentage)

        self.observation_space = spaces.Box(self.low_observation, self.high_observation, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        action_type = round(action[0])  # value in {0,1}
        position = math.floor(action[1])  # value in {0,...,self.number_of_points}
        x = action[2]
        y = action[3]

        if action_type == self.ADD_UPDATE:
            logging.debug("Setting coordinates for point %d to (%.2f, %.2f)", position, x*self.grid_size, y*self.grid_size)
            self.state[position] = (x*self.grid_size, y*self.grid_size)
        elif action_type == self.REMOVE:
            logging.debug("Removing coordinates for point %d", position)
            self.state[position] = (0, 0)

        reward, done = self.compute_step()

        # return observation, reward, done, info
        obs = [coordinate for tuple in self.state for coordinate in tuple]
        obs.append(0.5)
        return np.array(obs, dtype=np.float32), reward, done, {}

    def reset(self, seed: Optional[int] = None):
        #super().reset(seed=seed)
        # state is an empty sequence of points
        self.state = np.empty(self.number_of_points, dtype=object)
        for i in range(self.number_of_points):
            self.state[i] = (0, 0)  # (0,0) represents absence of information in the i-th cell
        return self.state

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

    def add_or_update_coordinates(self, index):
        old_coordinates = self.state[index]
        new_coordinates = (  # generate two completely random coordinates
            random.uniform(self.min_coordinate, self.max_coordinate),
            random.uniform(self.min_coordinate, self.max_coordinate)
        )
        # might as well generate new coordinates based on the (possibly) existing ones
        # if old_coordinates = (0, 0), generate new random coordinates
        # else new_coordinates = old_coordinates +- random_amount
        return new_coordinates

    def compute_step(self):
        done = False
        reward = 0
        road_points = self.get_road_points()
        logging.debug("Evaluating step. Current number of road points: %d (%s)", len(road_points), str(road_points))

        if len(road_points) < 3:  # cannot generate a good test (at most, a straight road with 2 points)
            reward = -10
        else:  # we should be able to generate a road with at least one turn
            the_test = RoadTestFactory.create_road_test(road_points)
            # check whether the road is a valid one
            is_valid, validation_message = self.beamng_executor.validate_test(the_test)
            if is_valid:
                # we run the test in the simulator
                test_outcome, description, execution_data = self.executor.execute_test(the_test)
            reward = 10
        return reward, done

    def get_road_points(self):
        road_points = np.array([], dtype=object)
        for i in range(self.number_of_points):
            if self.state[i][0] != 0 and self.state[i][1] != 0:
                np.insert(road_points, self.state[i])
        return road_points
