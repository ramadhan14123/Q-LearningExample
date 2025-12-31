from typing import Optional
import numpy as np

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        # break ties randomly among maxima
        q_row = self.Q[state]
        max_q = np.max(q_row)
        candidates = np.where(q_row == max_q)[0]
        return int(np.random.choice(candidates))

    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def set_epsilon(self, value: float):
        self.epsilon = max(self.epsilon_min, min(1.0, value))

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        Q = np.load(path)
        if Q.shape != self.Q.shape:
            raise ValueError("Loaded Q-table shape mismatch")
        self.Q = Q
