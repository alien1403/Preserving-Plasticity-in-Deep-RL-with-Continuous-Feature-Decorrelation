import matplotlib.pyplot as plt
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
    print(f"\n--- Starting Run: {'DQN + CovReg' if use_covreg else 'Baseline DQN'} ---")

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
        if step % 10_000 == 0:
            avg_r = np.mean(rewards_record[-10:]) if len(rewards_record) >= 10 else 0
            print(f"Step {step}/{total_steps} | Recent Return: {avg_r:.1f} | Feature Rank: {erank:.1f}")
    return steps_record, rewards_record, erank_record

def smooth(x, window=15):
    if len(x) < window: return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def main():
    steps_base, rew_base, erank_base = run_experiment(use_covreg=False)
    steps_cov, rew_cov, erank_cov = run_experiment(use_covreg=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(steps_base[:len(smooth(rew_base))], smooth(rew_base), label="Baseline DQN", color="#d62728", lw=2)
    ax1.plot(steps_cov[:len(smooth(rew_cov))], smooth(rew_cov), label="DQN + CovReg (Ours)", color="#1f77b4", lw=2)
    ax1.axvline(25_000, color="gray", linestyle="--", alpha=0.7, label="Phase 1 (Gravity Shift)")
    ax1.axvline(50_000, color="black", linestyle="--", alpha=0.7, label="Phase 2 (Inverted Actions)")
    ax1.set_title("Adaptation Performance (Episode Return)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Episode Return")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(steps_base[:len(smooth(erank_base))], smooth(erank_base), label="Baseline DQN", color="#d62728", lw=2)
    ax2.plot(steps_cov[:len(smooth(erank_cov))], smooth(erank_cov), label="DQN + CovReg (Ours)", color="#1f77b4", lw=2)
    ax2.axvline(25_000, color="gray", linestyle="--", alpha=0.7)
    ax2.axvline(50_000, color="black", linestyle="--", alpha=0.7)
    ax2.set_title("Representation Health (Effective Rank)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("SVD Effective Rank (Max = 128)")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=300)
    print("\n[Done] Benchmark complete! Output saved to results.png")

if __name__ == "__main__": main()
