import json
from typing import Dict, List, Tuple

class ConfigError(Exception):
    pass


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    required_top = ['width', 'height', 'start', 'goal', 'rewards']
    for k in required_top:
        if k not in cfg:
            raise ConfigError(f"Missing key '{k}' in config")

    width = cfg['width']
    height = cfg['height']
    start = tuple(cfg['start'])
    goal = tuple(cfg['goal'])
    walls = [tuple(w) for w in cfg.get('walls', [])]
    traps = [tuple(t) for t in cfg.get('traps', [])]

    rewards = cfg['rewards']
    for rk in ['goal', 'trap', 'wall', 'step']:
        if rk not in rewards:
            raise ConfigError(f"Missing rewards['{rk}'] in config")

    def in_bounds(p: Tuple[int, int]) -> bool:
        return 0 <= p[0] < width and 0 <= p[1] < height

    if not in_bounds(start):
        raise ConfigError("Start out of bounds")
    if not in_bounds(goal):
        raise ConfigError("Goal out of bounds")
    for w in walls:
        if not in_bounds(w):
            raise ConfigError(f"Wall {w} out of bounds")
    for t in traps:
        if not in_bounds(t):
            raise ConfigError(f"Trap {t} out of bounds")

    return {
        'width': width,
        'height': height,
        'start': start,
        'goal': goal,
        'walls': set(walls),
        'traps': set(traps),
        'rewards': rewards
    }
