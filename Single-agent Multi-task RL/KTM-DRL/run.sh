cd /home/yyq/Code/KTM-DRL/
conda activate ktm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yyq/.mujoco/mujoco200/bin

CUDA_VISIBLE_DEVICES=1 nohup python -u eval-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-eval.json --dir ./model/half --name EVA 2>&1 > evaluation.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u train-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-train.json --dir ./model/half --name TRN 2>&1 > training.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u train-ktm.py --seed 0 --cfg ./config/HALF/cfg-mt-half-train-cross_domain.json --dir ./model/half --name TRN 2>&1 > training_cross.out &