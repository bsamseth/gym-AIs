from agents.QLearningAgent import QLearningAgent


class RandomAgent(QLearningAgent):
    """
    The dumbest agent there is. Just produces a random move.
    """

    def __init__(self, action_space):
        super().__init__(None)
        self.is_trained = True

        class RandomModel(object):
            def predict(self, *args, **kwargs):
                return [action_space.sample()]

            def fit(self, *args, **kwargs):
                pass

            def load_weights(self, *args, **kwargs):
                pass

            def save_weights(self, *args, **kwargs):
                pass

        self.model = RandomModel()

    def build_model(self):  # Should override abstract method.
        pass


if __name__ == "__main__":
    from commons.GymRunner import GymRunner

    env_id = 'CartPole-v0'
    runner = GymRunner(env_id)
    agent = RandomAgent(runner.env.action_space)

    runner.run(agent, 1, render=True)

    agent.close()
    runner.close()