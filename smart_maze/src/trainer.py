import argparse
from typing import Dict, List
import numpy as np

from .config_loader import load_config
from .maze_env import MazeEnv
from .q_learning_agent import QLearningAgent


def train(env: MazeEnv,
          agent: QLearningAgent,
          episodes: int = 500,
          max_steps: int = 500,
          render_every: int = 0,
          visualizer=None) -> Dict:
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'successes': [],
    }

    vis = None
    if render_every > 0 and visualizer is not None:
        vis = visualizer(env.render_data(), cell_size=max(16, int(640 / max(env.width, env.height))))

    for ep in range(episodes):
        total_reward = 0.0
        steps = 0
        state = env.reset()
        done = False

        if vis and render_every and (ep % render_every == 0):
            vis.draw(env.pos)

        for t in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            if vis and render_every and (ep % render_every == 0):
                if not vis.process_events():
                    done = True
                    break
                vis.draw(env.pos)
            if done:
                break

        agent.decay_epsilon()
        success = 1 if env.pos == env.goal else 0
        metrics['episode_rewards'].append(total_reward)
        metrics['episode_steps'].append(steps)
        metrics['successes'].append(success)

    if vis:
        vis.close()

    return metrics


def plot_metrics(metrics: Dict, save_path: str = None, show: bool = False):
    import matplotlib.pyplot as plt
    rewards = metrics['episode_rewards']
    steps = metrics['episode_steps']
    successes = metrics['successes']

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    axs[0].plot(rewards)
    axs[0].set_title('Episode Reward')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')

    axs[1].plot(steps)
    axs[1].set_title('Episode Steps')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')

    # success rate rolling mean
    successes_arr = np.array(successes)
    window = max(1, len(successes)//20)
    if window > 1:
        kernel = np.ones(window)/window
        sr = np.convolve(successes_arr, kernel, mode='same')
    else:
        sr = successes_arr
    axs[2].plot(sr)
    axs[2].set_title('Success Rate (rolling)')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Rate')

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train Smart Maze agent (no graphics by default).')
    parser.add_argument('--config', type=str, required=True, help='Path to maze JSON config')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--render-every', type=int, default=0, help='Render every N episodes (requires pygame)')
    parser.add_argument('--show-plot', action='store_true')

    args = parser.parse_args()

    cfg = load_config(args.config)
    env = MazeEnv(cfg)
    agent = QLearningAgent(env.n_states, env.n_actions,
                           alpha=args.alpha, gamma=args.gamma,
                           epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay)

    # Optional visualization when render_every > 0
    vis_cls = None
    if args.render_every > 0:
        from .visualizer import Visualizer
        vis_cls = Visualizer

    metrics = train(env, agent, episodes=args.episodes, max_steps=args.max_steps,
                    render_every=args.render_every, visualizer=vis_cls)

    plot_metrics(metrics, save_path='smart_maze/assets/metrics.png', show=args.show_plot)

if __name__ == '__main__':
    main()
