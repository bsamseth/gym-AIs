from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from commons.GymRunner import GymRunner
from commons.QLearningAgent import QLearningAgent


class CartPoleAgent(QLearningAgent):
    """
    A Q-learning agent using a neural network to solve the
    CartPole environment in gym.

    Code modified from write-up on gym website.
    """
    def __init__(self, model_store_file=None):
        # Action space for CartPole is 2 (left force or right force).
        super().__init__(2, model_store_file)

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
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(Adam(lr=0.005), 'mse')

        return model

if __name__ == "__main__":
    runner = GymRunner('CartPole-v0')
    agent = CartPoleAgent('models/cartpole-v0.h5')

    runner.train(agent, 2000,
                 early_stopping=True,
                 early_stopping_mean_n=10,
                 early_stopping_mean_limit=195)
    runner.run(agent, 10, render=True)

    agent.close()
    runner.close()