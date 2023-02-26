# Solving the Rubik's Cube with Deep Reinforcement Learning

**By Justin Cheigh** <br>

In this project I attempt to solve the Rubik's Cube using [Deep Reinforcement Learning (Deep RL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning). This solver is a Tensorflow & Keras implementation of [DeepCube](https://openreview.net/pdf?id=Hyfn2jCcKm), a paper by McAleer et al. Some of the code is inspired by [azaharyan's](https://github.com/azaharyan) DeepCode [repository](https://github.com/azaharyan/DeepCube).

Essentially, a value/policy network is trained using Autodidactic Iteration (described in the aforementioned paper). After ADI we use the network to assist Monte Carlo Tree Search (MCTS) and solve the Cube.

To run, simply instantiate a Cube object (see cube.py) and run solve(cube) from main.py. 

For now, this project is completed. In reality, there are a lot of things that can be done. I initially wanted to create a long fully contained writeup with introductions to Reinforcement Learning, Artificial Neural Networks, Deep Reinforcement Learning, Monte Carlo Tree Search, Autodidactic Iteration, Group Theory, and certain key group theoretic results concerning the Rubik's Cube (i.e. the Fundamental Theorem of Cubology, Rubik's Cube as a semi-direct product, etc.). Other work can be done to speed up the training/solving process (most notably by using GPU computing). Finally, other design choices I would ideally implement is an argument parser to allow free choice of hyperparameters, more documentation, and an option to use a pretrained netwwork. However, I have lots of other cool projects in mind so will put this one on hold for a bit.

Began December 26th and finished (for now) February 26th. 
## Repository Layout 
```
├── README.md                              <- Project description/relevant links
├── adi.py                                 <- Autodidactic Iteration
├── cube.py                                <- Group theoretic Rubik's Cube class
├── cube_util.py                           <- Helper functions for cube.py
├── mcts.py                                <- Monte Carlo Tree Search 
├── node.py                                <- MCTS node class
├── model.py                               <- Value/policy network 
├── main.py                                <- Main file for training/searching
├── adi_util.py                            <- Helper functions for adi.py
``` 
