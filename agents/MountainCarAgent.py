import numpy as np

from agents.QLearningAgent import QLearningAgent


class MountainCarAgent(QLearningAgent):
    """
    A Q-learning agent using a human like approach to
    solve the MountainCar environment in gym.

    This is the current solution, as no network based network
    was able to solve this (yet) due to the very delayed reward,
    and time limit. Most variations never reach the top in time,
    and so is interpreted as bad. Need to stumble upon a solution,
    and then *really* exploit it.
    """

    def __init__(self):
        # Action space for MountainCar is 3 (left, nothing, right).
        super().__init__(action_space_size=3)
        self.is_trained = True

        class HumanDriver(object):
            """
            The drivers policy is simply to accelerate in which ever
            direction the car is currently moving.
            """
            def predict(self, state):
                pred = [0, 0, 0]
                x, v = state[0]
                if v > 0:
                    pred[2] = 1
                elif v < 0:
                    pred[0] = 1
                else:
                    pred[1] = 1
                return np.array([pred])

            def fit(self, *args, **kwargs):
                pass

            def load_weights(self, *args, **kwargs):
                pass

            def save_weights(self, *args, **kwargs):
                pass

        self.model = HumanDriver()

    def build_model(self):  # Should override abstract method.
        pass


if __name__ == "__main__":
    from commons.GymRunner import GymRunner

    env_id = 'MountainCar-v0'
    runner = GymRunner(env_id)
    agent = MountainCarAgent()

    # Make a runner for testing, with monitoring on (producing video).
    runner = GymRunner(env_id, monitor=True)
    runner.run(agent, 1, render=True)

    agent.close()
    runner.close()