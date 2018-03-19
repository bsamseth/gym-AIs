import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from commons.GymRunner import GymRunner
from agents.QLearningAgent import QLearningAgent


class FrozenLakeAgent(QLearningAgent):
    """
    A Q-learning agent using a neural network to solve the
    FrozenLake environment in gym.
    """

    def __init__(self, model_store_file=None):
        # Action space for FrozenLake is 4 (left, right, up and down).
        super().__init__(action_space_size=4,
                         model_store_file=model_store_file,
                         gamma=0.99,
                         epsilon=0.5,
                         epsilon_decay=0.995,
                         epsilon_min=0.01,
                         batch_size=10)

    def build_model(self):
        """
        Returns a Keras neural network model.

        We use a dense input layer, with relu, followed
        by a single hidden layer (also dense with relu),
        and finally a dense output layer with linear activation.
        :return: A Keras model object.
        """

        model = Sequential()

        # FrozenLake observation space is 1-dimensional.
        model.add(Dense(16, activation='relu', input_dim=1))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        return model

    def early_stopping(self, history):
        end_states, scores = np.asarray(history).T
        return np.mean(scores) > 0 and False


if __name__ == "__main__":
    env_id = 'FrozenLake-v0'
    runner = GymRunner(env_id)
    agent = FrozenLakeAgent('models/frozenlake-v0.h5')

    runner.train(agent, 2000, history_length=10)

    # Make a new runner for testing, with monitoring on (producing video).
    runner = GymRunner(env_id, monitor=True)
    runner.run(agent, 1, render=True)

    agent.close()
    runner.close()
