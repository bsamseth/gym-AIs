[![license](https://img.shields.io/badge/license-Unlicense-brightgreen.svg)](https://github.com/bsamseth/gym-AIs/blob/master/LICENSE)

# gym-AIs

This repository contains implementations of AIs (trying to) solve games from OpenAI Gym.
The code is not necessarily good, nor is all of it my original work. This is meant purely as a playground to
gain a better understanding of reinforcement learning.

## Environments

### CartPole

[CartPoleAgent.py](CartPoleAgent.py) has learned to solve the CartPole environment. Below is an example simulation.

![Example simulation of a model balancing a pole on a cart.](https://github.com/bsamseth/gym-AIs/blob/master/videos/cartpole.gif)

### MountainCar

[MountainCar.py](MountainCar.py) is a non-machine-learning solution to the MountainCar environment. A proper ML-based approach should
be made, but for now this is a *very* simple rule based policy driver. Below is an example simulation.

![Example simulation of a model balancing a pole on a cart.](https://github.com/bsamseth/gym-AIs/blob/master/videos/mountaincar.gif)
