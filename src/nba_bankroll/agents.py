"""Baselines and a compact optional PyTorch DQN."""
from collections import deque
import numpy as np
from .env import ACTIONS, OBS_SIZE


def baseline_action(env, name):
    if name == 'skip':
        return 0
    g = env.games[env.i]
    p = env.probability()
    sides = [('home', p, g.home_odds), ('away', 1-p, g.away_odds)]
    side, probability, odds = max(sides, key=lambda x: x[1]*x[2]-1)
    if probability * odds <= 1:
        return 0
    fraction = .02 if name == 'fixed' else .5 * (probability*odds-1)/(odds-1)
    # All policies share the same discrete action set and exposure constraints.
    candidates = [i for i, (s, f) in enumerate(ACTIONS)
                  if s == side and f <= min(.05, fraction) + 1e-9 and env.action_mask()[i]]
    return max(candidates, key=lambda i: ACTIONS[i][1]) if candidates else 0


class TabularQ:
    def __init__(self, seed=0, alpha=.1):
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.table = {}

    def key(self, obs):
        # Bankroll, locked funds, current drawdown, horizon, prices, estimated edge.
        return (int(np.clip(obs[1]*5, 0, 20)), int(obs[3]*20), int(obs[4]*10),
                int(obs[5]*4), int(obs[6]*10), int(obs[7]*10), int(obs[10]*10))

    def values(self, obs):
        return self.table.setdefault(self.key(obs), np.zeros(len(ACTIONS)))

    def act(self, obs, mask, epsilon=0.):
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return int(np.argmax(np.where(mask, self.values(obs), -np.inf)))

    def learn(self, obs, action, reward, nxt, done, mask):
        target = reward if done else reward + np.max(self.values(nxt)[mask])
        q = self.values(obs)
        q[action] += self.alpha * (target-q[action])


class DQN:
    def __init__(self, seed=0, learning_rate=3e-4, batch_size=64,
                 capacity=50000, warmup=256, target_interval=250):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError('DQN requires PyTorch. Install with: pip install -e ".[rl]"') from exc
        self.torch = torch
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = nn.Sequential(nn.Linear(OBS_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(ACTIONS)))
        import copy
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.buffer = deque(maxlen=capacity)
        self.batch_size, self.warmup = batch_size, max(warmup, batch_size)
        self.target_interval, self.updates = target_interval, 0

    def act(self, obs, mask, epsilon=0.):
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with self.torch.no_grad():
            q = self.online(self.torch.as_tensor(obs)).numpy()
        return int(np.argmax(np.where(mask, q, -np.inf)))

    def learn(self, obs, action, reward, nxt, done, mask):
        self.buffer.append((obs.copy(), action, reward, nxt.copy(), done, mask.copy()))
        if len(self.buffer) < self.warmup:
            return
        torch = self.torch
        indices = self.rng.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[int(i)] for i in indices]
        states, actions, rewards, next_states, dones, masks = map(np.asarray, zip(*batch))
        with torch.no_grad():
            # Double-DQN target, masked to exclude unaffordable next actions.
            next_q = self.online(torch.as_tensor(next_states))
            next_q[~torch.as_tensor(masks)] = -torch.inf
            best = next_q.argmax(1)
            future = self.target(torch.as_tensor(next_states)).gather(1, best[:, None]).squeeze(1)
            target = torch.as_tensor(rewards, dtype=torch.float32) + (~torch.as_tensor(dones)).float() * future
        predicted = self.online(torch.as_tensor(states)).gather(1, torch.as_tensor(actions)[:, None]).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
