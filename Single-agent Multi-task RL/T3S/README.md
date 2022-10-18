# T3S-MTRL-Pytorch

Implementation for "T3S: Improving Multi-Task Reinforcement Learning with Task-Specific Feature Selector and Scheduler". The source code can be found at https://github.com/yuyuanq/T3S-MTRL-Pytorch.

## Setup Environment

### Environement Requirements
* Python 3
* Pytorch 1.7
* posix_ipc
* tensorboardX
* tabulate
* gym
* MetaWorld (Please check next section to set-up MetaWorld)
* seaborn (for plotting)
* wandb (for logging)
### MetaWorld Setup
We evaluated our method on [MetaWorld](https://meta-world.github.io).

Since [MetaWorld](https://meta-world.github.io) is under active development, we perform all the experiment on our forked MetaWorld (https://github.com/RchalYang/metaworld).

```
# Our MetaWorld installation
git clone https://github.com/RchalYang/metaworld.git
cd metaworld
pip install -e .
```

## Our Network Structure and Task Scheduler

See ```HyperMTANPro``` in ```torchrl/hypernetworks/modules/hyper_mtanpro.py``` and ```TaskScheduler``` in ```torchrl/task_scheduler.py``` for details

## Training

All log and snapshot would be stored logging directory. Logging directory is default to be "./log/EXPERIMENT_NAME". 

EXPERIMENT_NAME can be set with "--id" argument when start experiment. And prefix directory can be set with "--log_dir" argument)

```
# MTSAC // MT10-FIXED
CUDA_VISIBLE_DEVICES=0 GROUP=MT10_MTSAC NAME=seed0 TASK_SAMPLE_NUM=10 nohup python -u starter/mt_para_mtsac.py --config meta_config/mt10/mtsac.json --id MT10_MTSAC --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MTSAC_0.out &

# MTSAC // MT10-RAND
CUDA_VISIBLE_DEVICES=0 GROUP=MT10_MTSAC_RAND NAME=seed0 TASK_SAMPLE_NUM=10 nohup python -u starter/mt_para_mtsac.py --config meta_config/mt10/mtsac_rand.json --id MT10_MTSAC_RAND --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_MTSAC_RAND_0.out &

# Soft Module // MT10-FIXED
CUDA_VISIBLE_DEVICES=0 GROUP=MT10_SM NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_mtsac_modular_gated_cas.py --config meta_config/mt10/modular_2_2_2_256_reweight.json --id MT10_Fixed_Modular_Shallow --seed 0 --worker_nums 10 --eval_worker_nums 10 2>&1 > nohup_outputs/MT10_Fixed_Modular_Shallow_0.out &

# Soft Module // MT10-RAND
CUDA_VISIBLE_DEVICES=0 GROUP=MT10_SM_RAND NAME=seed0 TASK_SAMPLE_NUM=10 nohup python starter/mt_para_mtsac_modular_gated_cas.py --config meta_config/mt10/modular_2_2_2_256_reweight_rand.json --id MT10_Fixed_Modular_Shallow_RAND --seed 0 --worker_nums 10 --eval_worker_nums 10 2>&1 > nohup_outputs/MT10_Fixed_Modular_Shallow_RAND_0.out &

# T3S // MT10-FIXED
CUDA_VISIBLE_DEVICES=0 GROUP=MT10_T3S_k5 NAME=seed0 TASK_SAMPLE_NUM=5 nohup python starter/mt_para_hypersac.py --config meta_config/mt10/mtsac.json --id MT10_MMOE_k5 --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_T3S_k5_0.out &

# T3S // MT10-RAND
CUDA_VISIBLE_DEVICES=0 GROUP=MT10_T3S_RAND_k5 NAME=seed0 TASK_SAMPLE_NUM=5 nohup python starter/mt_para_hypersac.py --config meta_config/mt10/mtsac_rand.json --id MT10_MMOE_RAND_k5 --worker_nums 10 --eval_worker_nums 10 --seed 0 2>&1 > nohup_outputs/MT10_T3S_RAND_k5_0.out &

```

## Plot Training Curve

To plot the training curves, you could use the following command.

* id argument is used for multiple experiment names.

* seed argument is used for multiple seeds

* replace "mean_success_rate" with different entry to see different curve for different entry.

```
python torchrl/utils/plot_csv.py --id EXPERIMENTS --env_name mt10 --entry "mean_success_rate" --add_tag POSTFIX_FOR_OUTPUT_FILES --seed SEEDS
```

