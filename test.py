import imageio
import argparse
import gym
from agent import PolicyGradientAgent

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='./output_images/LunarLander.gif')
parser.add_argument('--checkpoint_dir', type=str, default='./PGtrain/Policy.pth')
parser.add_argument('--save_video', type=bool, default=True)
parser.add_argument('--fps', type=int, default=30)

args = parser.parse_args()

def main():
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = PolicyGradientAgent(args.checkpoint_dir)

    agent.network.eval()
    agent.load_models()

    video = imageio.get_writer(args.filename, fps=args.fps)

    done = False
    state = env.reset()
    if len(state) > 1:
        state, _ = state

    done = False
    while not done:
        action, _ = agent.sample(state)
        state, _, done, _, _ = env.step(action)

        if args.save_video:
            video.append_data(env.render())


if __name__ == '__main__':
    main()