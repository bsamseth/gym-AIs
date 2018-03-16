import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from commons.GymRunner import GymRunner
from commons.QLearningAgent import QLearningAgent


class CartPoleAgent(QLearningAgent):
    """
    A Q-learning agent using a neural network to solve the
    CartPole environment in gym.

    Code modified from write-up on gym website by github.com/ruippeixotog.
    """

    def __init__(self, model_store_file=None):
        # Action space for CartPole is 2 (left force or right force).
        super().__init__(action_space_size=2,
                         model_store_file=model_store_file,
                         gamma=0.99,
                         epsilon=0.5,
                         epsilon_decay=0.995,
                         epsilon_min=0.01,
                         batch_size=32)

    def build_model(self):
        """
        Returns a Keras neural network model.

        We use a dense input layer, with relu, followed
        by a single hidden layer (also dense with relu),
        and finally a dense output layer with linear activation.
        :return: A Keras model object.
        """

        model = Sequential()

        # CartPole observation space is four-dimensional.
        model.add(Dense(16, activation='relu', input_dim=4))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        return model

    def early_stopping(self, history):
        end_states, scores = np.asarray(history).T
        return np.mean(scores) > 199


if __name__ == "__main__":
    runner = GymRunner('CartPole-v0')
    agent = CartPoleAgent('models/cartpole-v0.h5')

    runner.train(agent, 2000, history_length=10)
    runner.run(agent, 10, render=True)

    agent.close()
    runner.close()
