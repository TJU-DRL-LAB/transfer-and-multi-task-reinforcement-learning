## Installation
Ubuntu 16.04

Python 3.7.4

Torch 1.10.1

Cuda 11.3

Other packages requirements are in requirements.txt

The method is tested on SMAC(https://github.com/oxwhirl/smac).

## command
### Train a policy on a task from scratch
For Marines
```
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=api_vdn_multihead_relation_v33 --env-config=sc2 with env_args.map_name=5m obs_agent_id=False epsilon_anneal_time=100000 td_lambda=0.6 obs_last_action=False batch_size_run=8 runner=parallel  buffer_size=5000 t_max=5000000  batch_size=128 debug_dir=False run_type=train checkpoint_path="" save_path="" seed=0 is_curriculum=False cof=1 lr=0.001 aggregation=sum wandb_name=5m
```
For S & Z and MMM
```
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=api_vdn_multihead_relation_v33_dividehyper --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z obs_agent_id=False epsilon_anneal_time=100000 td_lambda=0.6 obs_last_action=False runner=parallel  buffer_size=5000 batch_size_run=4 t_max=10000000  batch_size=128 debug_dir=False run_type=train checkpoint_path="" save_path="" seed=0 is_curriculum=False cof=1 lr=0.001 aggregation=sum wandb_name=3s5z_vs_3s6z
```
### Transfer learning
For Marines
```
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=api_vdn_multihead_relation_v33_reload --env-config=sc2 with env_args.map_name=5m_vs_6m obs_agent_id=False obs_last_action=False runner=parallel batch_size_run=8 buffer_size=5000 t_max=10000000 epsilon_anneal_time=500000 batch_size=128 td_lambda=0.6 debug_dir=False run_type=train checkpoint_path="model_path" save_path="" seed=0 is_curriculum=False cof=1 lr=0.0005 aggregation=sum wandb_name=6m_vs_8m
```
For S & Z and MMM
```
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=api_vdn_multihead_relation_v33_dividehyper_reload --env-config=sc2 with env_args.map_name=3s5z_vs_4s8z obs_agent_id=False epsilon_anneal_time=500000 td_lambda=0.6 obs_last_action=False runner=parallel buffer_size=5000 batch_size_run=4 t_max=10000000 batch_size=128 debug_dir=False run_type=train checkpoint_path="model_path" save_path= seed=0 is_curriculum=True cof=1 lr=0.0005 aggregation=sum wandb_name=3s5z_vs_4s8z
```
### Task selection criterion
Use the script below to get task difficulty and task similarity criterion. All configurations can be seen in the comments of the scripts.

Task difficulty
```
./run_evaluate.sh
```
Task similarity
```
./run_collect.sh
```
With the criteria, select appropriate task as next curriculum and do curriculum transfer using the transfer learning command.

For our compared baselines HPN-QMIX and DYMA, they can be run with 
```
--config=api_qmix_multihead
```
and
```
--config=dyan_attackunit_vdn
```
```
--config=dyan_attackunit_vdn_reload
```
**All commands are run in directory: code/ma_perm**
