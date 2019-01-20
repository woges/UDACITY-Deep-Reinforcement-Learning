# Project 1: Navigation
![Part2_Project_Navigation](./img/Trained_agent_banana_env_PER_Dueling_DDQN_V01.gif)

## Overview

In this project an agent is set up and trained to navigate and collect yellow bananas in a large square world.

## Dependencies

If you have already installed all the necessary dependencies for the **navigation project** in part 2 you should be good to go! If not, you should install them to get started on this project => [Getting Started for Part 2 Navigation Project](../Part2_How_to_get_started). 
 
## Basic Build Instructions

1. Clone or fork this repository.
2. Install all necessary dependencies
3. Launch the Jupyter notebook: `jupyter notebook`
4. Change Kernel to `drlnd`
5. Select `Navigation_PER_Duelling_DDQN_V01.ipynb`
6. Follow the instructions in the notebook or execute the code cells you are interested in.
 
Note that cells may depend on previous cells. The notebook explains clearly what each code cell does.

## Goal of this project

The goal of this project is to design, train and evaluate an agent that collects as many yellow bananas as possible in a large square world while avoiding to pick up blue bananas. Therefore a deep reinforcement algorithm has to be implemented. Here an angent with an Deep Q-Learning (DQN) algorithm and several additional improvements like Double DQN (DDQN), Duelling Neural Architecture and DQN with prioritized experience replay is implemented.

## Project Environment Details 

Here we will use Unity's rich environments to design, train, and evaluate deep reinforcement learning algorithms. **Unity Machine Learning Agents ([ML-Agents](https://github.com/Unity-Technologies/ml-agents))** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

### Note

The project environment is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

### Environment

The state space has 37 dimensions and contains a ray-based perception of objects around the agent's forward direction.

- 7 rays are pointing from the agent at the following angles: [20, 45, 70, 90, 110, 135, 160] 
    - Note that 90 degree is pointing directly in front of the agent
    - Each ray can encounter one of four detectable objects along with a  distance measure: [Yellow Banana, Wall, Blue Banana, Agent, Distance]

It also contains the agent's velocity:
- Left/Right
- Forward/Backward

So as a result we get:
 **7 rays * [4 detectable objects + 1 distance] + 2 velocities = 37states**

 Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward.
- `1` - move backward.
- `2` - turn left.
- `3` - turn right.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. 

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Results

See [Report.md](./report.md) for more details.

## Literature

[Deep Q-Networks](./resources/2_001_2015_Mnih_et_al_Human-level_control_through_DRL_DQNNaturePaper.pdf)

## Contributing

No further updates nor contributions are requested.  This project is static.

## License

Part2_Project_Navigation results are released under the [MIT License](./LICENSE)