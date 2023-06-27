#!/bin/sh
env="overcooked"
scenario="counter_circuit_o_1order"
num_agents=2
algo="mappo" #"rmappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents}  --seed ${seed} --horizon 2000 --hidden_size 32 --N_layer 2 --max_grad_norm 0.1 --layer_N 2 \
    --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 6 --mini_batch_size 2000 --episode_length 400 --num_env_steps 80000000 --entropy_coef 0.1 --gae_lambda 0.98 --clip_param 0.05  \
    --ppo_epoch 8 --gain 0.01 --lr 8e-4 --critic_lr 8e-4 --wandb_name "xxx" --user_name "yuchao" --use_eval --n_eval_rollout_threads 5 --eval_interval 25 --num_reward_shaping_steps 6500000 \

done