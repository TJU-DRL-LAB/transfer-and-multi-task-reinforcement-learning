# Cross-Domain Adaptive Transfer Reinforcement Learning Based on State-Action Correspondence

This repository contains the source code of [Cross-Domain Adaptive Transfer Reinforcement Learning Based on State-Action Correspondence]([Cross-domain Adaptive Transfer Reinforcement Learning Based on State-Action Correspondence | OpenReview](https://openreview.net/forum?id=ShN3hPUsce5)) accepted on UAI 2022. 

## Introduction

Despite the impressive success achieved in various domains, deep reinforcement learning (DRL) is still faced with the sample inefficiency problem. Transfer learning (TL), which leverages prior knowledge from different but related tasks to accelerate the target task learning, has emerged as a promising direction to improve RL efficiency. 

The majority of prior work considers TL across tasks with the same state-action spaces, while transferring across domains with different state-action spaces is relatively unexplored. Furthermore, such existing cross-domain transfer approaches only enable transfer from a single source policy, leaving open the important question of how to best transfer from multiple source policies. 

This paper proposes a novel framework called **Cross-domain Adaptive Transfer (CAT)** to accelerate DRL. CAT learns the state-action correspondence from each source task to the target task and adaptively transfers knowledge from multiple source task policies to the target policy. CAT can be easily combined with existing DRL algorithms and experimental results show that CAT significantly accelerates learning and outperforms other cross-domain transfer methods on multiple continuous action control tasks.

A brief overview is illustrated below.

![CAT_00.png](https://s2.loli.net/2022/06/06/lpJjAaGKVOutHx9.png)

In order to extract more useful knowledge, we propose four properties that the state embeddings should satisfy:

- The embeddings should be **task-aligned** to maximize the cumulative discount rewards in the target MDP. 
- The input states and state embeddings should be **highly correlated** so that the agent can receive the most appropriate guidance from source policies in the current state.
- The embeddings should preserve enough information about the source task so that $\phi_i(s)$ can be reconstructed to the target task as **consistently** as possible.
- In addition to the correspondence on the single state, $s_s$ and $s_t$, the state embedding should keep the **correspondence** between state sequences of the source and target tasks.

CAT contains three components: **Correction Module**, **Self-Adaptation Module** and **Agent Module**:

- The correction module is used to learn state embeddings that can satisfy properties (3-4) and to learn action embeddings that can better capture the semantics of actions of the source and target tasks.
- The self-adaptation module determines when and which source policy should be transferred by evaluating the source policies and generating weights for different source policies.
- The agent module allows our agent to distill knowledge from source policies, select actions to execute in the target environment, and learn a high-performing policy.

## Experimental Results

Experimental results show that CAT significantly accelerates RL and outperforms other cross-domain transfer methods.

![results.jpg](https://s2.loli.net/2022/06/16/DpSEYRIMr7vxu1s.png)

## Dependencies

- Python 3.6+
- tensorflow 1.12.0
- gym 0.10.5
- mujoco-py 2.0.2.8
- numpy 1.19.5
- click 7.0
- num2words 0.5.10
- scipy 1.3.3

## Usage

To run CAT, for example:

```eval
python -m baselines.run --env=CentipedeEight-v1 --source_env=CentipedeFour-v1 --source_env1=CentipedeSix-v1 --alg=ppo2 --num_timesteps=2e6 --pi_scope=student0 --vf_scope=vf_student0 --network=mlp --teacher_network=mlp --save_path=models --save_filenumber=0 --save_interval=10 --log_path=logs --teacher_paths teacher_models/centipede_four/0/00970 --teacher_paths1 teacher_models/centipede_six/0/00970 --mapping=learned
```

will run transfer from `CentipedeFour` and `CentipedeSix ` to `CentipedeEight` using the corresponding teacher models which you can also train by yourself using the following command:

```
python -m baselines.run --env=CentipedeFour-v1 --alg=ppo2 --num_timesteps=2e6 --network=mlp --save_path=teacher_models/centipede_four/5 --save_interval=10 --log_path=teacher_models/centipede_four/5
```

 If you want to expand to more source policies like transfer from `CentipedeFour`, `CentipedeSix` and `CentipedeEight` to `CentipedeTen`, you should add the corresponding state encoders, reverse state encoders and action encoders in the code and use the following command:

```
python -m baselines.run --env=CentipedeTen-v1 --source_env=CentipedeFour-v1 --source_env1=CentipedeSix-v1 --source_env2=CentipedeEight-v1 --alg=ppo2 --num_timesteps=2e6 --pi_scope=student0 --vf_scope=vf_student0 --network=mlp --teacher_network=mlp --save_path=models --save_filenumber=0 --save_interval=10 --log_path=logs --teacher_paths teacher_models/centipede_four/0/00970 --teacher_paths1 teacher_models/centipede_six/0/00970 --teacher_paths2 teacher_models/centipede_eight/0/00970 --mapping=learned
```



## Citation

If you found this work to be useful in your own research, please considering citing the following:

```
@article{You_545,
  title={Cross-domain adaptive transfer reinforcement learning based on state-action correspondence},
  author={Heng, You and Tianpei, Yang and Yan, Zheng and Jianye, HAO and Matthew, E. Taylor},
  year={2022}
}
```

