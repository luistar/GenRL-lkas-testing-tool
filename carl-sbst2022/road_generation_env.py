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
        """
        Converts the internal representation of road points (in self.state) into a list of points that can be processed
        by the test executor.
        """
        pass

    def compute_step(self):
        done = False
        reward = 0
        execution_data = []
        max_oob_percentage = 0
        road_points = self.get_road_points()
        logging.debug("Evaluating step. Current number of road points: %d (%s)", len(road_points), str(road_points))

        if len(road_points) < 3:  # cannot generate a good test (at most, a straight road with 2 points)
            logging.debug("Test with less than 3 points. Negative reward.")
            reward = -10
        else:  # we should be able to generate a road with at least one turn
            the_test = RoadTestFactory.create_road_test(road_points)
            # check whether the road is a valid one
            is_valid, validation_message = self.executor.validate_test(the_test)
            if is_valid:
                logging.debug("Test seems valid")
                # we run the test in the simulator
                test_outcome, description, execution_data = self.executor.execute_test(the_test)
                logging.debug(f"Simulation results: {test_outcome}, {description}, {execution_data}")
                if test_outcome == "ERROR":
                    # Could not simulate the test case. Probably the test is malformed test and evaded preliminary validation.
                    logging.debug("Test seemed valid, but test outcome was ERROR. Negative reward.")
                    reward = -10  # give same reward as invalid test case
                elif test_outcome == "PASS":
                    # Test is valid, and passed. Compute reward based on execution data
                    reward = self.compute_reward(execution_data)
                    logging.debug(f"Test is valid and passed. Reward was {reward}, with {max_oob_percentage} OOB.")
                    max_oob_percentage = self.get_max_oob_percentage(execution_data)
                elif test_outcome == "FAIL":
                    reward = 100
                    max_oob_percentage = self.get_max_oob_percentage(execution_data)
                    logging.debug(f"Test is valid and failed. Reward was {reward}, with {max_oob_percentage} OOB.")
                    # todo save current test
            else:
                logging.debug(f"Test is invalid: {validation_message}")
        return reward, max_oob_percentage

    def compute_reward(self, execution_data):
        reward = pow(execution_data[0].max_oob_percentage/10, 2)
        return reward

    @staticmethod
    def get_max_oob_percentage(execution_data):
        """
        execution_data is a list of SimulationDataRecord (which is a named tuple).
        We iterate over each record, and get the max oob percentage.
        """
        max_oob_percentage = 0
        for record in execution_data:
            if record.max_oob_percentage > max_oob_percentage:
                max_oob_percentage = record.max_oob_percentage
        return max_oob_percentage
