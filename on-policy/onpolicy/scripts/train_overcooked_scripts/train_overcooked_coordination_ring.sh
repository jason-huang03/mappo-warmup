#!/bin/sh
env="overcooked"
scenario="coordination_ring"
num_agents=2
algo="mappo" #"rmappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents}  --seed ${seed} --horizon 2000 --hidden_size 256 \
    --n_training_threads 1 --n_rollout_threads 40 --num_mini_batch 1 --episode_length 400 --num_env_steps 8000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "jason-huang03" --use_eval --n_eval_rollout_threads 5 --eval_interval 5 
done