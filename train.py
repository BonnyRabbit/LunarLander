import numpy as np
import torch
import gym
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import PolicyGradientAgent

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--figure_file', type=str, default='./output_images/rewards')
parser.add_argument('--checkpoint_dir', type=str, default='./PGtrain/Policy.pth')

args = parser.parse_args()

def main():
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = PolicyGradientAgent(args.checkpoint_dir)

    # train
    agent.network.train()
    EPISODE_PER_BATCH = 5

    avg_total_rewards, avg_final_rewards = [], []
    prg_bar = tqdm(range(args.max_episodes))
    for batch in prg_bar:

        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []

        for episode in range(EPISODE_PER_BATCH):

            state = env.reset()
            if len(state) > 1:
                state, _ = state

            total_reward, total_step = 0, 0

            while True:

                action, log_prob = agent.sample(state)
                next_state, reward, done, _, _ = env.step(action)

                log_probs.append(log_prob)
                state = next_state
                total_reward += reward
                total_step += 1
                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    rewards.append(np.full(total_step, total_reward))
                    break

        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

        rewards = np.concatenate(rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards + 1e-9))
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

        if (batch+1) % 100 == 0:
            agent.save_models()

    plt.plot(avg_total_rewards)
    plt.title("Total Rewards")
    plt.show()

    plt.plot(avg_final_rewards)
    plt.title("Final Rewards")
    plt.show()

if __name__ == '__main__':
    main()