# Project 3: Contiuous Control
![Part3_Project_Continuous_Control](./img/Continuous_Control_trained_agent_action.gif)

## Overview

In this project an agent is set up and trained to move a double-jointed arm to target locations. The goal of the agent is to maintain its position at the moving target location for as many time steps as possible.

## Dependencies

If you have already installed all the necessary dependencies for the **continous control project** in part 3 you should be good to go! If not, you should install them to get started on this project => [Getting Started for Part 3 Continuous Control Project](../Part3_How_to_get_started). 
 
## Basic Build Instructions

1. Clone or fork this repository.
2. Install all necessary dependencies
3. Launch the Jupyter notebook: `jupyter notebook`
4. Change Kernel to `drlnd`
5. Select `./results/Part3_Project_Contiuous_Control.ipynb`
6. Follow the instructions in the notebook or execute the code cells you are interested in.
 
Note that cells may depend on previous cells. The notebook explains clearly what each code cell does.

## Goal of this project

The goal of this project is to design, train and evaluate an agent that moves a double-jointed arm to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

Therefore a deep reinforcement algorithm has to be implemented. Here an angent with an Deep Deterministic Policy Gradient (DDPG) algorithm and several additional improvements for getting a more stabilized learning is implemented:

•   Replay buffer
•   Target Q network with soft target updates
•   Batch normalization
•   Noise added, so we can treat the problem of exploration independent form the learning algorithm (Ornstein-Uhlenbeck process)

## Project Environment Details 

Here we will use Unity's rich environments to design, train, and evaluate deep reinforcement learning algorithms. **Unity Machine Learning Agents ([ML-Agents](https://github.com/Unity-Technologies/ml-agents))** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

### Note

The project environment is similar to, but not identical to the reacher environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

### Environment


- Set-up: Double-jointed arm which can move to target locations.
- Goal: The agents must move its hand to the goal location, and keep it there.
-  Agents: The first version contains a single agent.
-  Agents: The second version contains 20 identical agents, each with its own copy of the environment.

The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

- Agent Reward Function (independent):
    - +0.1 Each step agent's hand is in goal location.
- Brains: One Brain with the following observation/action space.
    - Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
        Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
- Reset Parameters: Two, corresponding to goal size, and goal movement speed.
- Benchmark Mean Reward: 30

Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The task is episodic, and in order to solve the environment:
- the agent must get an average score of +30 over 100 consecutive episodes for the first version.
- The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).

### Results

See [Report.md](./Report.md) for more details.

## Literature

[DDPG](./resources/305_20160229_Lillicrap_et_al_Continuous control with DRL.pdf)

## Contributing

No further updates nor contributions are requested.  This project is static.

## License

Part3_Continuous_Control results are released under the [MIT License](./LICENSE)