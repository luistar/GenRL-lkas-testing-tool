import numpy as np
import math
import logging as log
import matplotlib.pyplot as plt

from code_pipeline.tests_generation import RoadTestFactory


class CustomTestGenerator:
    """
        Generates a single test based on a sequence of points, possibly generated from
        https://jsfiddle.net/jgf6keax/2/show
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):
        log.info("Starting test generation")

        road_points = []

        # insert the points from https://jsfiddle.net/jgf6keax/2/show

        # the following four road points trigger a failure
        road_points.append((95.80959468674862, 75.01687198931195))
        road_points.append((26.272263077684364, 108.62671169506721))
        road_points.append((72.81917499387318, 152.11169216481326))
        road_points.append((155.13927252524255, 104.21557515386719))


        # Creating the RoadTest from the points
        the_test = RoadTestFactory.create_road_test(road_points)

        # Send the test for execution
        test_outcome, description, execution_data = self.executor.execute_test(the_test)

        # Plot the OOB_Percentage: How much the car is outside the road?
        oob_percentage = [state.oob_percentage for state in execution_data]
        log.info("Collected %d states information. Max is %.3f", len(oob_percentage), max(oob_percentage))

        plt.figure()
        plt.plot(oob_percentage, 'bo')
        plt.show()

        # Print test outcome
        log.info("test_outcome %s", test_outcome)
        log.info("description %s", description)

        import time
        time.sleep(10)