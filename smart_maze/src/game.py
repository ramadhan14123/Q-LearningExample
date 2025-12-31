import argparse
from .config_loader import load_config
from .maze_env import MazeEnv
from .q_learning_agent import QLearningAgent
from .trainer import train, plot_metrics


def main():
    parser = argparse.ArgumentParser(description='Run Smart Maze with visualization.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--cell-size', type=int, default=32)
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')
    parser.add_argument('--show-plot', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = MazeEnv(cfg)
    agent = QLearningAgent(env.n_states, env.n_actions,
                           alpha=args.alpha, gamma=args.gamma,
                           epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay)

    vis = None
    render_every = 1 if args.render else 0
    if args.render:
        # Lazy import to avoid requiring pygame when not rendering
        from .visualizer import Visualizer
        vis = Visualizer(env.render_data(), cell_size=args.cell_size)

    # Training with visualization every episode
    metrics = train(env, agent, episodes=args.episodes, max_steps=args.max_steps,
                    render_every=render_every,
                    visualizer=(lambda data, cell_size: vis) if vis else None)

    plot_metrics(metrics, save_path='smart_maze/assets/metrics.png', show=args.show_plot)

    if vis:
        vis.close()

if __name__ == '__main__':
    main()
