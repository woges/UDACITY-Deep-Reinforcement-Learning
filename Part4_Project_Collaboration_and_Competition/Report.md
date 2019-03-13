# Project 4: Collaboration and Competition


## Goal of this project

The goal of this project is to design, train and evaluate two agents that moves a double-jointed arm to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

Therefore a deep reinforcement algorithm has to be implemented. Here an angent with an Deep Deterministic Policy Gradient (DDPG) algorithm and several additional improvements for getting a more stabilized learning is implemented:

-   Replay buffer
-   Target Q network with soft target updates
-   Batch normalization
-   Noise added, so we can treat the problem of exploration independent form the learning algorithm (Ornstein-Uhlenbeck process)

## Description of the implementation

### Learning algorithm

This project implements an off-policy method called Deep Deterministic Policy Gradient and described in the paper [Continuous control with deep reinforcement learning](./resources/305_20160229_Lillicrap_et_al_Continuous_control_with_DRL.pdf). Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

Unfortunately, reinforcement learning is notoriously unstable when neural networks are used to represent the action values. Therefore we should use **two key** features to overcome with this and enable RL agents to converge, more reliably during training:

- **Experience Replay**
    => use of a rolling history of the past data via replay pool. The act of sampling a small batch of tuples form the replay buffer in oder to learn is known as **experience replay**. Advantages:
    - the behavior distribution is averaged over many of its previous states
    - smoothing out learning and
    - avoiding oscillations
    - each step of the experience is potentially used in many weight updates
    - more efficient use of observed experiences
    - breaks up the potentially highly correlated sequence of experienced tupels

- **Fixed Q-Targets**
    => use of a target network to represent the old Q-function, which will be used to compute the loss of every action during training
    -  otherwise as the Q-functions values change at each step of training the value estimates can easily spiral out of control
    -  To use the fixed Q-Targets technique, you need a second set of parameters w- which you can initialize to w. 

![Fixed Q-Targets](./img/Fixed_Q_Targets.png)

**Pyeudo Code for Deep Deterministic Policy Gradient (DDPG) algorithm**
![Deep Deterministic Policy Gradient (DDPG) algorithm](./img/DDPG.png)

This project implements a Policy Based method called [DDPG](./resources/305_20160229_Lillicrap_et_al_Continuous_control_with_DRL.pdf).


## Training and Hyperparameter

The DDPG agent uses the following 
PARAMETER VALUES:
- BUFFER_SIZE = int(1e6)    # replay buffer size
- BATCH_SIZE = 256          # minibatch size
- RANDOM_SEED = 2           # ramdom seed
- GAMMA = 0.99              # discount factor
- TAU = 1e-2                # for soft update of target parameters
- LR_ACTOR = 1e-3           # learning rate of the actor
- LR_CRITIC = 1e-3          # learning rate of the critic
- WEIGHT_DECAY = 0          # L2 weight decay
- NUM_AGENTS = 2            # Number of agents
- LEARN_EVERY = 2          # Learn every x time steps
- LEARN_UPDATES = 4         # Number of learning steps 

MODEL: 
- A_FC1_UNITS = 256         # Actor: Number of nodes in first hidden layer
- A_FC2_UNITS = 128         # Actor: Number of nodes in second hidden layer
- C_FCS1_UNITS = 256        # Critic: Number of nodes in first hidden layer
- C_FC2_UNITS = 128         # Critic: Number of nodes in second hidden layer
- OPTIMIZER = ADAM 

NOISE PARAMETERS:
- OUNOISE_Theta = 0.15      # Theata for Ornstein-Uhlenbeck process
- OUNOISE_SIGMA = 0.2       # Sigma for Ornstein-Uhlenbeck process
- OUNOISE_MU = 0.           # Mue for Ornstein-Uhlenbeck process
- NOISE_DEC = 0.9999        # Reductioni rate for noise

TRAINING:
- N_EPISODES = 500          # Number of episodes
- MAX_T = 5000              # Max length of one episode
- BATCH_NORMAL = True       # Enable batch normalization


## Results

The agent is able to receive an average reward (over 100 episodes) of at least +0.5 in only 72 episodes as shown in the following chart.  

![Result](./results/DDPG_tennis_trained_performance.png)

### Untrained agent performing random actions

![Part4_Project_Collaboration_and_Competition](./img/Continuous_Control_random_action.gif)

### Trained agent performing appropriate actions - average score 0.5

![Part4_Project_Collaboration_and_Competition](./img/Continuous_Control_trained_agent_action.gif)

### Trained agent performing appropriate actions - average score 2.5

![Part4_Project_Collaboration_and_Competition](./img/Continuous_Control_trained_agent_action.gif)

### Future Ideas for improving the agent's performance

For the objective of this project, the model produced a satisfactory result. However, the agents are not stable or reliable on-goingly. The resuls vary considerably from episode to episode. Even worse, letting the agents continue learning, sometimes the scores drop to 0 again and never recover.

One possible good improvement for the multi agent environments could be done by implementing the Twin Delayed DDPG (TD3) as proposed von OpenAI's [Spinning Up](https://spinningup.openai.com/en/latest/index.html) website. 

`While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and other kinds of tuning. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function. Twin Delayed DDPG (TD3) is an algorithm which addresses this issue by introducing three critical tricks:`
`- Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.
 - Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.
 - Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.`

`Together, these three tricks result in substantially improved performance over baseline DDPG.`

Pseudocode:

![Part4_Project_Collaboration_and_Competition](./img/TD3_openai.svg)


## Literature

[DDPG](https://arxiv.org/abs/1509.02971)

[TRPO](https://arxiv.org/abs/1604.06778)

[PPO](https://arxiv.org/pdf/1707.06347.pdf)

[PPO@openai.com](https://blog.openai.com/openai-baselines-ppo/)

[D4PG](https://openreview.net/forum?id=SyZipzbCb)

[A3C](https://arxiv.org/pdf/1602.01783.pdf)


    Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., et al. Continuous control with deep reinforcement learning. arXiv.org, 2015.
    Lowe, R., WU, Y., Tamar, A., Harb, J., Abbeel, P., and Mordatch, I. Multi- agent actor-critic for mixed cooperative-competitive environments. 2017.
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal Policy Optimization Algorithms. arXiv.org, 2017.


## Contributing

No further updates nor contributions are requested.  This project is static.

## License

Part4_Collaboration_and_Competition results are released under the [MIT License](./LICENSE)