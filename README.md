# Distributed-Distributional-DrQ
Distributed Distributional DrQ is a model-free and off-policy RL algorithm for continuous control tasks based on the image-vision input. This is an actor-critic method with data-augmentation and D4PG as the backbone.

Code implementation of the [Distributed Distributional DrQ][paper] algorithm in Pytorch.

Distributed Distributional method is based on the D4PG original paper: [[ArXiv]](https://arxiv.org/abs/1804.08617).

Part code based on original DrQ-v2 implementation code from the facebook AI research:
[[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning]](https://arxiv.org/abs/2107.09645)

Implementation code of D4PG reference from [[D4PG-pytorch]](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter14/06_train_d4pg.py)

<p align="center">
  <img width="22%" src="https://imgur.com/O5Va3NY.gif">
  <img width="22%" src="https://imgur.com/PCOR9Mm.gif">
  <img width="22%" src="https://imgur.com/H0ab6tz.gif"> </p>


## Method

Combine D4PG with DrQ-v2

## Instruction

Tasks are based on the MuJoCo, install [MuJoCo](http://www.mujoco.org/) 

## License
Soft DrQ-v2 is licensed under the MIT license.

