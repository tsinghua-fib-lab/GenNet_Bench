import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0")


def generate_w(num_prefence, reward_size, fixed_w=None):
    if fixed_w is not None:
        w = np.random.rand(num_prefence - 1, reward_size)
        
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence - 1, 1)
        return np.concatenate((fixed_w, w))
    else:
        w = np.random.rand(num_prefence, reward_size)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w


def renew_w(preferences, dim):
    w = np.random.rand(len(preferences[0]))
    w = np.abs(w) / np.linalg.norm(w, ord=1, axis=0)
    preferences[dim] = w
    return preferences


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action, config):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)], config)


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action, config)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,
                                  config=config
                                  )
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp
                                  for acsp in env.action_space],
                                 config.dim_obj
                                 )
    t = 0

    fixed_w = None
    explore_w = generate_w(1, config.dim_obj, fixed_w)  

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("\nEpisodes %i-%i of %i" % (ep_i + 1,
                                          ep_i + 1 + config.n_rollout_threads,
                                          config.n_episodes))
        obs = env.reset()
        
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(
            config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            
            torch_obs = torch.Tensor(obs)
            
            torch_agent_actions = maddpg.step(torch_obs, torch.Tensor(explore_w), explore=True)
            
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            
            
            actions = [ac for ac in agent_actions]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                update_w = generate_w(config.num_w_sample, config.dim_obj, explore_w)  
                update_w = torch.Tensor(update_w).to(device)
                if USE_CUDA:
                    maddpg.prep_training(device=device)
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA, device=device)
                        maddpg.update(sample, a_i, update_w, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
                  config.episode_length * config.n_rollout_threads)
        explore_w = renew_w(explore_w, 0)  
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="wireless", type=str, help="Name of environment")
    parser.add_argument("--model_name", default="maddpg", type=str,
                        help="Wireless Optimize " +
                             "model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=1, type=int)
    parser.add_argument("--dim_obj", default=3, type=int, help="目标维度")
    parser.add_argument("--num_w_sample", default=5, type=int, help="偏好采样数量")

    parser.add_argument("--buffer_length", default=int(10), type=int)
    parser.add_argument("--n_episodes", default=5000, type=int, help="多少个episode")
    parser.add_argument("--episode_length", default=10, type=int, help="一个episode多少时间步")
    parser.add_argument("--steps_per_update", default=10, type=int, help="多少时间步更新一次")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=5000, type=int,
                        help="需要和n_episodes一致，探索率衰减的episode数")
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=10, type=int, help="多少episode保存一次模型")
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
