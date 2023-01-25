# Solving the Rubik's Cube with Deep Reinforcement Learning

**By Justin Cheigh** <br>

In this project I attempt to solve the Rubik's Cube using [Deep Reinforcement Learning (Deep RL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning). This solver is an Tensorflow & Keras implementation of [DeepCube](https://openreview.net/pdf?id=Hyfn2jCcKm), a paper by McAleer et al. The code is inspired by [azaharyan's](https://github.com/azaharyan) DeepCode [repository](https://github.com/azaharyan/DeepCube).

I also created a fully contained writeup that includes introductions to Reinforcement Learning, Artificial Neural Networks, Deep Reinforcement Learning, Monte Carlo Tree Search, Autodidactic Iteration, Group Theory, and certain key group theoretic results concerning the Rubik's Cube (i.e. the Fundamental Theorem of Cubology, Rubik's Cube as a semi-direct product, etc.). 


Essentially, a value/policy network is trained using Autodidactic Iteration (described in the aforementioned paper). After ADI we use the network to assist Monte Carlo Tree Search (MCTS) and solve the Cube.

## Repository Layout 
```
├── README.md                              <- Project description/relevant links
├── adi.py                                 <- Autodidactic Iteration
├── cube.py                                <- Group theoretic Rubik's Cube class
├── mcts.py                                <- Monte Carlo Tree Search 
├── model.py                               <- Value/policy network
├── main.py                                <- Main file for training/searching
├── utils                                  <- Contains .py utility files 
``` 
