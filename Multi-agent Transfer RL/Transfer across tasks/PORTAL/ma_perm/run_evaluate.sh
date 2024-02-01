policy_list=('results_opensource/curriculum/MAACL/sandz/0/1s4z_vs_5s1z/model/1s4z_vs_5s1z_api_vdn_mh')
save_path="results_opensource/curriculum/MAACL/sandz"
#policy_list=('results_opensource/curriculum/ours_sz1/0/2s3z/model/2s3z_api_vdn_multihead_relation_v33_dividehyper_2million')
#save_path="results_opensource/curriculum/ours_sz1"
env_list=('2s3z_vs_2s4z 3s5z_vs_3s6z 3s5z_vs_3s7z 3s5z_vs_4s6z 1s4z_vs_6s1z 3s5z_vs_8s2z 3s5z_vs_4s7z 2s3z_vs_2s5z 3s5z_vs_4s8z')
seed=0
#policy='2s3z'
policy='1s4z_vs_5s1z'
#config='api_vdn_multihead_relation_v33_dividehyper_reload'
config='api_vdn_multihead_reload'

for p in ${policy_list[*]}
do
  for e in  ${env_list[*]}
    do
      CUDA_VISIBLE_DEVICES="3" python -u src/main_load_debug.py --config="$config" --env-config=sc2 with env_args.map_name="$e" obs_agent_id=False obs_last_action=False runner=episode batch_size_run=1 buffer_size=1000 t_max=10050000 epsilon_anneal_time=100000 batch_size=32 td_lambda=0.6 run_type=evaluate debug_dir=True checkpoint_path="$p" save_path="$save_path" seed="$seed" is_curriculum=False cof=1 lr=0.001 wandb_name=test aggregation=sum
    done
done