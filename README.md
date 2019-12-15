# Reinforcement-Learning

This repository contains the implementation of different Reinforcement Learning Algorithms I have learned throughout the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 

Each algorithm is used to solve different [OpenAI gym environements](http://gym.openai.com/envs/).

## Algorithms

1. [Monte Carlo](./monte-carlo/)
2. [Temporal Difference Algorithms](./temporal-difference/)
   * [Sarsa(0)](./temporal-difference/CliffWalking/)
   * Q-Learning / Sarsamax
      * [CliffWalking-v0 environment](./temporal-difference/CliffWalking/)
      * [Taxi-v2 environment](./temporal-difference/Taxi-V2/)
   * [Expected Sarsa](./temporal-difference/CliffWalking/)
3. [RL in Continuous Space](./RL-in-continuous-space/)
   * [Discretization](./RL-in-continuous-space/Discretization.ipynb)
   * [Tile Coding](./RL-in-continuous-space/Tile_Coding.ipynb)

## Projects

1. [Navigation](https://github.com/anubhavshrimal/Navigation_Udacity_DRLND_P1): Train an agent to collect yellow bananas while avoiding blue bananas using `Deep Q-learning Algorithm`.
2. [Continuous Control](https://github.com/anubhavshrimal/Continuous_Control_Udacity_DRLND_P2): Train an robotic arm to reach target locations using `DDPG Algorithm`.
3. [Collaboration and Competition](https://github.com/anubhavshrimal/Collaboration_Competition_Udacity_DRLND_P3): Train a pair of agents to play tennis using `MADDPG Algorithm`.

## Dependency Installation

Follow the instructions given in the [Installation_Guide.md](Installation_Guide.md) to install the dependencies and run the code present in this repository locally.

## RL Notes

I have also provided the notes I created while learning the above mentioned algorithms and techniques. You can find these notes in the [RL-Notes](./RL-Notes/) folder.

1. [The RL Framework](./RL-Notes/01&#32;-&#32;The&#32;RL&#32;framework.pdf)
2. [Monte Carlo Methods](./RL-Notes/02&#32;-&#32;Monte&#32;Carlo&#32;Methods.pdf)
3. [Temporal Difference Methods](./RL-Notes/03&#32;-&#32;Temporal&#32;Difference&#32;Methods.pdf)
4. [RL in Continuous Space](./RL-Notes/04&#32;-&#32;RL&#32;in&#32;Continuous&#32;Space.pdf)
5. [Cheatsheet](./RL-Notes/cheatsheet.pdf)