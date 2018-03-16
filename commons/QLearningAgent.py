import abc
from collections import deque
import numpy as np
import random
import os


class QLearningAgent(object):
    """
    Abstract base class for a Q-learning agent.

    This class provided the expected interface between classes
    such as GymRunner, and the underlying model.

    Subclasses may only need to supply an implementation of the
    build_model() method, defining a specific model. It may also be of
    interest to tune some of the hyper-parameters.

    Code modified from write-up on gym website by github.com/ruippeixotog.
    """

    def __init__(self, action_space_size, model_store_file=None,
                 gamma=0.9, epsilon=1, epsilon_decay=0.995,
                 epsilon_min=0.05, batch_size=32,
                 memory_size_multiplier=5):
        """
        Initialize the agent.

        If model_store_file is provided and it is a valid file, weights
        will be loaded from this file, and the agent will be marked as trained.

        If model_store_file is provided, regardless of its prior existence,
        the model will be saved to this filename when the agent is closed.

        :param action_space_size: number of actions that can be made in a state
        :param model_store_file: file used to save model weights in
        :param gamma: discount rate for future rewards
        :param epsilon: probability of choosing random actions in training
        :param epsilon_decay: decay rate applied per training batch
        :param epsilon_min:  minimum value epsilon may decay to
        :param batch_size: size of training batches sampled when learning
        :param memory_size_multiplier: memory size, units of batch_size
        """
        self.action_space_size = action_space_size
        self.model_store_file = model_store_file
        self.is_trained = False

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Container for observed simulations.
        self.memory = deque(maxlen=self.batch_size * memory_size_multiplier)

        self.model = self.build_model()

        # Initialize model weights if a valid file has been provided.
        if self.model_store_file and os.path.isfile(self.model_store_file):
            self.model.load_weights(self.model_store_file)
            self.is_trained = True

    @abc.abstractmethod
    def build_model(self):
        """
        :return: A Keras-type model.
        """
        pass

    def early_stopping(self, history):
        """
        This method can be used to determine if we should stop training early,
        based on the history of how the training has gone.

        Default is to return False, indicating that we should not stop
        :param history: sequence of (final_state, score) pairs
        :return: True if we should terminate training, False otherwise.
        """
        return False

    def select_action(self, state, training=False):
        """
        Produce an action based on the learned model.

        If training is true, we might chose a random action.
        This is done with a probability of self.epsilon.

        :param state: the state to produce an action from
        :param training: true if we should include randomness
        :return: An action to take, as a non-negative integer.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_space_size)
        return np.argmax(self.model.predict(state)[0])

    def record(self, state, action, reward, next_state, done):
        """
        Save the given simulation result in memory for later training.

        :param state: the state in which we produced
        :param action: this action,
        :param reward: which gave this reward
        :param next_state: and produced this next state
        :param done: which meant the simulation was done (true/false)
        """
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """
        Sample a random subset of the recorded simulations, and train the
        model to better fit its response accordingly.
        """
        batch_size = min(self.batch_size, len(self.memory))

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            # The target Q-value, given by Bellman's equation:
            # Q(S, A) = R + discount * max( Q(S', A') for A' in descendant S' of S )
            if done:
                target = reward
            else:
                target = reward + \
                         self.gamma * np.max(self.model.predict(next_state)[0])

            # What model produces now:
            target_response = self.model.predict(state)

            # Modify value for action with the Bellman target:
            target_response[0][action] = target

            # Fit model to produce target_response when in state:
            self.model.fit(state, target_response, epochs=1, verbose=0)

        # Decay epsilon.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def close(self):
        """
        If self.model_store_file is set, save the model weights.
        """
        if self.model_store_file:
            self.model.save_weights(self.model_store_file, overwrite=True)
