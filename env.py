import gymnasium as gym

class NonStationaryCartPole:
    def __init__(self, switch_interval: int = 25_000, seed: int = 42):
        self.env = gym.make("CartPole-v1")
        self.switch_interval = switch_interval
        self.seed = seed
        self.total_steps = 0
        self.current_phase = 0

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        return obs

    def step(self, action: int):
        self.total_steps += 1
        new_phase = self.total_steps // self.switch_interval
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self._apply_phase_shift(self.current_phase)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

    def _apply_phase_shift(self, phase: int):
        unwrapped = self.env.unwrapped
        if phase == 1:
            unwrapped.gravity = 24.5
            unwrapped.length = 0.25
            unwrapped.masspole = 0.2

    @property
    def observation_space(self): return self.env.observation_space
    @property
    def action_space(self): return self.env.action_space
