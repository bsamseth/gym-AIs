import numpy as np

from commons.GymRunner import GymRunner
from commons.QLearningAgent import QLearningAgent


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

    def __init__(self, model_store_file=None):
        # Action space for MountainCar is 3 (left, nothing, right).
        super().__init__(action_space_size=3)
        self.is_trained = True

    def build_model(self):
        """
        Returns a human logical driver model.

        The drivers policy is simply to accelerate in which ever
        direction the car i currently moving.

        :return: A "model" with same interface as a Keras-model.
        """

        class HumanDriver(object):
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

        return HumanDriver()


if __name__ == "__main__":
    env_id = 'MountainCar-v0'
    runner = GymRunner(env_id)
    agent = MountainCarAgent('models/mountaincar-v0.h5')

    # Training is skipped, as model cannot learn.
    runner.train(agent, 2000, history_length=10)

    # Make a new runner for testing, with monitoring on (producing video).
    runner = GymRunner(env_id, monitor=True)
    runner.run(agent, 1, render=True)

    agent.close()
    runner.close()