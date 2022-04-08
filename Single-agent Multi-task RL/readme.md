# Multi-task RL
One of the directions to improve the **efficiency of the RL agent** is by multi-tasking-based learning. One of the well-known definitions states that “Multi-task Learning is an approach to inductive transfer that improves generalization by using the domain information contained in the training signals of related tasks as an inductive bias”.

This repo contains representative research works of on the topic of Multi-task RL. 
<div align=center><img align="center" src="assets/multi-task intro.png" alt="multi-task_intro" style="zoom:20%;" /></div>

## An Overall View of Research Works in This Repo

This repo include two parts:

1. Current and previous influential work in Multi-task RL research field, which are common used baselines presented in recent papers. Some implementation are citing from @mtrl
2. Multi-task RL research works from TJU-RL-Lab.
Both will be constantly updated to include new researches. (The development of this repo is in progress at present.)

## Installation

The algorithms in this repo are all implemented **python 3.6** (and versions above). **PyTorch** are the main DL code frameworks we adopt in this repo with different choices in different algorithms.

Note that the algorithms contained in this repo may not use all the same environments. Please check the README of specific algorithms for detailed installation guidance.

## TODO
- [ ] update our work

## Related Work

Here we provide a useful list of representative related works on Multi-task RL.

### Network-Architecture:

- **Soft-Modularization** - https://arxiv.org/abs/2003.13661v2 

### Gradient-based:
- **GradDrop** -  [https://arxiv.org/abs/2010.06808](https://arxiv.org/abs/2010.06808) 
- **PCGrad** - [https://arxiv.org/abs/2001.06782](https://arxiv.org/abs/2001.06782) 
- **CAGrad** - [https://arxiv.org/abs/2110.14048](https://arxiv.org/abs/2110.14048) 

### Others:

+ **CARE** -  https://arxiv.org/abs/2102.06177
