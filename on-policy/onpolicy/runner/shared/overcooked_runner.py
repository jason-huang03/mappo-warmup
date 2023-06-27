import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import pdb
from tqdm import tqdm

## TODO: modify the overcooked environment


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunner(Runner):
    def __init__(self, config):
        super(OvercookedRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()

        num_env_steps = 0

        # determine how many episodes to run
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            pbar = tqdm(range(self.episode_length))
            total_reward = []
            total_success = []
            for step in range(self.episode_length):
                # sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)


                # observe reward and next obs
              
                obs, rewards, dones, infos = self.envs.step(actions)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                total_reward.append(rewards)
                total_success.append(rewards >= 19)

                num_env_steps += self.n_rollout_threads

                self.insert(data)
                pbar.update(1)

                pbar.set_description(
                    "Total Episode {}".format(episodes)
                    + " | Episode {}".format(episode)
                    + " | Training Reward: {}".format(np.mean(np.sum(total_reward, axis=0)))
                    + " | Training Success: {}".format(np.mean(np.sum(total_success, axis=0)))
                    + " | env_step: {}".format(num_env_steps)
                )

            pbar.close()

            self.compute()
            train_infos = self.train()

            # eval
            if (episode + 1) % self.eval_interval == 0 and self.use_eval:
                self.eval()

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        
        

        # init the share_obs and obs at zero th step
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """
        :return values: np.ndarray, (num_rollout_threads, num_agents, 1)
        """
        self.trainer.prep_rollout()

        ## this combine the num_rollout_threads and the num_agents together
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )

        ## split the data back to num_rollout_threads and num_agents
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        # ? overcooked does not use rnn?
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )

        # ? what is the use of masks? what is its relation with dones?
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    def eval(self):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        pbar = tqdm(range(self.episode_length))
        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()

            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )

            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions
            )
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

            pbar.update(1)
            pbar.set_description("evaluating")

        pbar.close()

        eval_episode_rewards = np.array(eval_episode_rewards, dtype=np.float32)
        eval_average_episode_rewards = np.mean(
            np.sum(eval_episode_rewards, axis=0)
        ).item()
        print(
            "eval_average_episode_rewards: ",
            eval_average_episode_rewards,
            "successfully delivered soup: ",
            eval_average_episode_rewards / 20.0,
        )

    def render(self):
        raise NotImplementedError
