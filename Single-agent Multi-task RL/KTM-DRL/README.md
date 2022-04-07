

# Knowledge Transfer in Multi-Task Deep Reinforcement Learning for Continuous Control

This repository reproduces the work in [Knowledge Transfer in Multi-Task Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/2010.07494) accepted on NeurIPS 2020, which is built on the official repository [KTM-DRL](https://github.com/xuzhiyuan1528/KTM-DRL). The official repository does not support training on benchmark B (cross_domain setting) due to the lack of teacher models, traning code and training required files. In addition, there are also some problems when configuring the virtual environment according to the readme file in the official repository. We solve the mentioned problems as follows:

- We provide a note when configuring the virtual environment in **Dependencies** to avoid errors you may encounter.
- We provide teacher models trained by ourselves in [models](https://pan.baidu.com/s/1nnm_6u59xGyYyGjiftj_HA) and training required files on both benchmark A and benchmark B.
- We evalute and show our reproduced results compared with the experimental results in the paper as shown in the following **Experimental Results**. Specific commands are provided in **Evaluation** and **Training**.

## Introduction

Despite the impressive performance of DRL on individual tasks, it remains challenging to train a single DRL agent to undertake multiple different tasks. Unlike single-task DRL, which learns a control policy for an individual task, multi-task DRL requires an agent to learn a single control policy that could perform well on multiple different tasks. A straightforward approach is to directly train a DRL agent for multiple tasks one by one using a traditional single-task learning algorithm, which has been shown to deliver poor performance and may even fail on some tasks, due to differences and possible interference among multiple tasks. An effective approach is to tackle this problem with knowledge transfer, e.g., Actor-Mimic and policy distillation. These methods usually aimed at training a single multi-task agent under the guidance of task-specifific teachers. However, these methods were designed based on DQN for discrete control tasks.

This paper presents a **Knowledge Transfer** based **Multi-task Deep Reinforcement Learning** framework **(KTM-DRL)** for continuous control, which enables a single DRL agent to achieve expert-level performance on multiple different tasks by learning from task-specific teachers. 

A brief overview is illustrated below.

![KTM-DRL.png](https://s2.loli.net/2022/03/26/alfOT6MtEPs3zBV.png)

KTM-DRL consists of two learning stages: **the offline knowledge transfer stage** and **the online learning stage**:

- The multi-task agent leverages an **offline knowledge transfer** algorithm designed particularly for the actor-critic architecture to quickly learn a control policy from the experience of task-specific teachers. 
- Then, under the guidance of these knowledgeable teachers, the agent further improves itself by learning from new transition samples collected during **online learning**.

## Experimental Results

The experimental results in the paper are roughly consistent with our reproduced results and are shown below:

**HalfCheetah task group (Benchmark A):**

| Method       | KTM-DRL   | Ideal     | KTM-DRL (ours) | Ideal (ours) |
| ------------ | --------- | --------- | -------------- | ------------ |
| HCSmallTorso | **10348** | 8743      | **10802**      | 9276         |
| HCBigTorso   | **10364** | 9067      | **10823**      | 9549         |
| HCSmallLeg   | **10594** | 9575      | **11232**      | 10796        |
| HCBigLeg     | 10402     | **10683** | 10803          | **10968**    |
| HCSmallFoot  | 8836      | **9633**  | 9713           | **10332**    |
| HCBigFoot    | **9239**  | 8902      | **10324**      | 10187        |
| HCSmallThigh | 10470     | **10769** | **11115**      | 10897        |
| HCBigThigh   | **9787**  | 9524      | **10014**      | 8824         |

**Cross-domain task group (Benchmark B):**

| Method        | KTM-DRL  | Ideal     | KTM-DRL (ours) | Ideal (ours) |
| ------------- | -------- | --------- | -------------- | ------------ |
| Ant           | 5836     | **5839**  | **5503**       | 5061         |
| Hopper        | 3565     | **3588**  | 3065           | **3219**     |
| Walker2d      | **4863** | 4797      | 3847           | **4040**     |
| HalfCheetah   | 10921    | **10969** | 9780           | **10471**    |
| InvPendulum   | **1000** | 1000      | **1000**       | 1000         |
| InvDbPendulum | 9347     | **9351**  | 9308           | **9309**     |

## Dependencies

- Python 3.7.7 (Optional, also works with older versions)
- [PyTorch 1.5.1](https://github.com/pytorch/pytorch)
- [MuJoCo 200](http://www.mujoco.org/index.html)
- [mujoco-py 2.0.2.5](https://github.com/openai/mujoco-py)
- [OpenAI Gym 0.17.2](https://github.com/openai/gym)
- [gym-extensions](https://github.com/Breakend/gym-extensions) (Optional, only for HalfCheetah task group)

**Note:** The environment may report an error, you need to replace all "v0" with "v1" in gym_extensions/continuous/mujoco/init.py to fix this.

## Evaluation

To use the pre-trained models of task-specific teachers and multi-task agent, download from the [Dropbox Link](https://pan.baidu.com/s/1nnm_6u59xGyYyGjiftj_HA) (KTM0) and put the `half` folder into `./model/` or `/your_own_path/`. 


To evaluate the multi-task agent and its corresponding task-specific teachers, run this command for **HalfCheetah task group (Benchmark A)**:

```eval
python3 eval-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-eval.json --dir ./model/half --name EVA
```

Run this command for **Benchmark B**:

```
python3 eval-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-train-cross_domain.json --dir ./model/half --name EVA
```

## Training

To train the model (s) in the paper, run this command for **HalfCheetah task group (Benchmark A)**:

```train
python3 train-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-train.json --dir ./model/half --name TRN
```

Run this command for **Benchmark B**:

```
python3 train-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-train-cross_domain.json --dir ./model/half --name TRN
```

