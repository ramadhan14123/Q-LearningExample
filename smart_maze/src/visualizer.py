import pygame
from typing import Tuple, Set, Dict

COLORS = {
    'bg': (240, 240, 240),
    'grid': (200, 200, 200),
    'wall': (60, 60, 60),
    'trap': (200, 80, 80),
    'goal': (80, 200, 80),
    'start': (80, 80, 200),
    'agent': (30, 144, 255),
}

class Visualizer:
    def __init__(self, env_data: Dict, cell_size: int = 32):
        pygame.init()
        self.cell_size = cell_size
        self.width = env_data['width']
        self.height = env_data['height']
        self.surface = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
        pygame.display.set_caption('Smart Maze')
        self.clock = pygame.time.Clock()
        self.env_data = env_data

    def draw(self, agent_pos: Tuple[int, int]):
        self.surface.fill(COLORS['bg'])
        # Grid
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.surface, COLORS['grid'], rect, width=1)
        # Walls
        for (x, y) in self.env_data['walls']:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.surface, COLORS['wall'], rect)
        # Traps
        for (x, y) in self.env_data['traps']:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.surface, COLORS['trap'], rect)
        # Goal
        gx, gy = self.env_data['goal']
        grect = pygame.Rect(gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.surface, COLORS['goal'], grect)
        # Start
        sx, sy = self.env_data['start']
        srect = pygame.Rect(sx * self.cell_size, sy * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.surface, COLORS['start'], srect)
        # Agent
        ax, ay = agent_pos
        arect = pygame.Rect(ax * self.cell_size, ay * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.surface, COLORS['agent'], arect)

        pygame.display.flip()
        self.clock.tick(60)

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        pygame.quit()
