policy_list=('results_opensource/curriculum/ours_MMM/0/MMM5/model/MMM5_api_vdn_mh_aid=0_na=12')
#env_list=('MMM7 MMM11 MMM12 MMM8')
#policy_list=('results_opensource/curriculum/ours_sz1/0/3s5z_vs_3s6z/model/3s5z_vs_3s6z_api_vdn_mh_aid=0_na=8_5million')
#env_list=('3s5z_vs_4s7z 3s5z_vs_8s2z 3s6z_vs_4s8z 3s5z_vs_4s8z')
#policy_list=('results_opensource/curriculum/ours_sz1/0/2s3z/model/2s3z_api_vdn_multihead_relation_v33_dividehyper_2million')
env_list=('')
seed=0
# dont include seed in the path
save_path="results_opensource/curriculum/ours_MMM"
policy='MMM5'
target_env='MMM10'
#config='api_vdn_multihead_reload'
#config='api_vdn_multihead_relation_v33_reload'
config='api_vdn_multihead_relation_v33_dividehyper_reload'

# it's fixed
n=10000

for p in ${policy_list[*]}
do
  for e in  ${env_list[*]}
    do
          CUDA_VISIBLE_DEVICES="3" python -u src/main_load_debug.py --config="$config" --env-config=sc2 with env_args.map_name="$e" obs_agent_id=False obs_last_action=False runner=episode batch_size_run=1 buffer_size=100 t_max=10050000 epsilon_anneal_time=100000 batch_size=32 td_lambda=0.6 debug_dir=True run_type=collect aggregation=sum checkpoint_path="$p" save_path="$save_path" seed="$seed" is_curriculum=False cof=1 lr=0.001 wandb_name=test
    done
done
