from collections import deque
import numpy as np
import gym


class GymRunner(object):
    """
    Class used as main interface between an agent and the gym environment.

    Code modified from write-up on gym website by github.com/ruippeixotog.

    Example usage:

        runner = GymRunner('CartPole-v0')
        agent = ...

        runner.train(agent, 2000,
                    early_stopping=True,
                    early_stopping_mean_n=10,
                    early_stopping_mean_limit=195)
        runner.run(agent, 10, render=True)

        runner.close()
    """

    def __init__(self, env_id, max_timesteps=10000,
                 monitor=False, logging_dir='log/'):
        """
        :param env_id: name of the gym environment to use
        :param max_timesteps: maximum timesteps per episode
        :param monitor: true if a monitor wrapper should be used
        :param logging_dir: directory for monitor output
        """
        self.max_timesteps = max_timesteps
        self.env = gym.make(env_id)
        if monitor:
            self.env = gym.wrappers.Monitor(self.env,
                                            logging_dir,
                                            force=True)

    def train(self, agent, *args, **kwargs):
        """
        Wrapper for GymRunner.run, enabling training if needed.
        If the agent has been marked with agent.is_trained == True,
        no training will be performed. All agents will be marked
        with agent.is_trained = True after a call to this method.

        :param agent: agent to train
        :param args: positional arguments to be passed to run
        :param kwargs: keyword ---""---
        """
        if not agent.is_trained:
            kwargs['training'] = True
            self.run(agent, *args, **kwargs)
            agent.is_trained = True

    def run(self, agent, num_episodes,
            training=False,
            verbose=True,
            render=False,
            history_length=10):
        """
        Simulate a given number of episodes, using the agent to select
        actions along the way.

        If training is set to true, then the agent will learn from the results.

        If render is set to true, each episode will be rendered to screen.

        :param agent: agent used for action selection
        :param num_episodes: maximum number of episodes to simulate
        :param training: true if agent should be trained, false by default
        :param verbose: true will print info about each episode, true by default
        :param render: true if the simulation should be rendered on screen
        :param history_length: number of episodes to include in the history
                               sent to the agent.early_stopping during training
        """

        # A sequence which keeps only the n most recent entries.
        # Used for early stopping.
        history = deque(maxlen=history_length)

        for episode in range(num_episodes):
            # Get ready for a new episode. Ensure dimensions are OK for agent.
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])

            # Accumulator for the total score for this episode.
            total_reward = 0

            for _ in range(self.max_timesteps):
                if render:
                    self.env.render()

                action = agent.select_action(state, training)

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])

                if training:
                    agent.record(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:  # Environment says we are done, should be respected.
                    break

            history.append((state, total_reward))

            if verbose:
                print(f'Episode {episode + 1}/{num_episodes} ' +
                      f'score = {total_reward}, ' +
                      (f'eps = {agent.epsilon:4.3f}, ' if training else '') +
                      f'mean {np.mean(np.asarray(history)[:,1])}')

            # Determine if we should stop training early.
            if training and agent.early_stopping(history):
                break

            # Let model learn from its recorded memory.
            if training:
                agent.learn()

    def close(self):
        """
        Close the environment for proper exit from rendered graphics.
        """
        self.env.close()
