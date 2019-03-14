# Project 4: Collaboration and Competition
![Part4_Project_Collaboration_and_Competition](./img/Collaboration_and_Competition_Tennis_trained.gif)

## Overview

In this environment, two agents control rackets to bounce a ball over a net. The goal of the agents is to keep the ball in play for as many time steps as possible.

## Dependencies

If you have already installed all the necessary dependencies for the **collaboration and competition project** in part 4 you should be good to go! If not, you should install them to get started on this project => [Getting Started for Part 4 Collaboration and Competition Project](../Part4_How_to_get_started). 
 
## Basic Build Instructions

1. Clone or fork this repository.
2. Install all necessary dependencies
3. Launch the Jupyter notebook: `jupyter notebook`
4. Change Kernel to `drlnd`
5. Select `./results/Part4_Project_Collaboration_Competition.ipynb`
6. Follow the instructions in the notebook or execute the code cells you are interested in.
 
Note that cells may depend on previous cells. The notebook explains clearly what each code cell does.

## Goal of this project

The goal of this project is to design, train and evaluate two agents so that they can bounce a ball over a net as often as possible. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

Therefore a deep reinforcement algorithm has to be implemented. Here an multi agent concept with an Deep Deterministic Policy Gradient (DDPG) algorithm and several additional improvements for getting a more stabilized learning is implemented:

-   Replay buffer
-   Target Q network with soft target updates
-   Batch normalization
-   Noise added, so we can treat the problem of exploration independent form the learning algorithm (Ornstein-Uhlenbeck process)

## Project Environment Details 

Here we will use Unity's rich environments to design, train, and evaluate deep reinforcement learning algorithms. **Unity Machine Learning Agents ([ML-Agents](https://github.com/Unity-Technologies/ml-agents))** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

### Note

The project environment is similar to, but not identical to the tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

### Environment


- Set-up: agents which bounce a ball over a net.
- Goal: the agents must keep the ball in play.
- Agent Reward Function (independent):
    - +0.1 if an agent hits the ball over the net.
    - -0.01 if an agent lets a ball hit the ground
    - -0.01 if an agent hits the ball out of bounds
- Brains: One Brain with the following observation/action space.
    - observation space: 8 variables corresponding to position and velocity of the ball and racket.
        - each agent receives its own, local observations
    - action space: (continuous) size of 2, 
        - corresponding to movement toward or away from the net
        - jumping

Thus, the goal of the agents is to maintain the ball in play for as many time steps as possible. The task is episodic, and in order to solve the environment:
- the agent must get an average score of 0.5 over 100 consecutive episodes, after taking the maximum over both agents.
Specifically:
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
- This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.
- The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Results

See [Report.md](./Report.md) for more details.

## Literature

[Continuous control with deep reinforcement learning - DDPG](https://arxiv.org/abs/1509.02971)

[Deep Reinforcement Learning for Continuous Control - TRPO](https://arxiv.org/abs/1604.06778)

[Multi- agent actor-critic for mixed cooperative-competitive environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)

## Contributing

No further updates nor contributions are requested.  This project is static.

## License

Part4_Collaboration_and_Competition results are released under the [MIT License](./LICENSE)