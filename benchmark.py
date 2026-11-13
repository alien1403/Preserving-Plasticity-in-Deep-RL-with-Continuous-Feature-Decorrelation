import numpy as np
from dqn import DQNAgent
from env import NonStationaryCartPole

def run_experiment(use_covreg: bool, total_steps: int = 75_000, seed: int = 42):
    env = NonStationaryCartPole(switch_interval=25_000, seed=seed)
    agent = DQNAgent(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n, use_covreg=use_covreg, covreg_coeff=0.08)
    obs = env.reset()
    ep_reward = 0.0
    steps_record, rewards_record, erank_record = [], [], []
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 20_000

    for step in range(1, total_steps + 1):
        eps = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * step / epsilon_decay)
        action = agent.select_action(obs, eps)
        next_obs, reward, done, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward
        _, _, erank = agent.update()
        if step % 200 == 0: agent.update_target_network()
        if done:
            obs = env.reset()
            steps_record.append(step)
            rewards_record.append(ep_reward)
            erank_record.append(erank if erank > 0 else 1.0)
            ep_reward = 0.0
    return steps_record, rewards_record, erank_record
