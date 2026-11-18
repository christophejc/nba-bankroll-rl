"""Finite-season replay with cash locked until recorded settlement times."""
import math
import numpy as np

# One side per game; fractions of equity at decision time.
ACTIONS = [(None, 0.0)] + [(side, f) for side in ('home', 'away') for f in (.01, .02, .05)]
OBS_SIZE = 11


class BankrollEnv:
    def __init__(self, games, initial_bankroll=1000., risk_penalty=0.,
                 exposure_limit=.20, bankroll_floor=.20, elo_probs=None, elo_weight=0.):
        if not games or len({g.season for g in games}) != 1:
            raise ValueError('An episode must contain exactly one nonempty season')
        if not 0 <= elo_weight <= 1:
            raise ValueError('elo_weight must be in [0, 1]')
        if initial_bankroll <= 0 or risk_penalty < 0:
            raise ValueError('Invalid bankroll or risk penalty')
        if not 0 < exposure_limit <= 1 or not 0 < bankroll_floor < 1:
            raise ValueError('Invalid exposure limit or bankroll floor')
        self.games = sorted(games, key=lambda g: (g.decision_time, g.game_id))
        self.initial = initial_bankroll
        self.risk_penalty = risk_penalty
        self.exposure_limit = exposure_limit
        self.floor = bankroll_floor
        self.elo_probs = elo_probs or {}
        self.elo_weight = elo_weight

    @property
    def locked(self):
        return sum(x[1] for x in self.pending)

    @property
    def equity(self):
        # Outstanding stakes valued at cost, not at unknowable final payoff.
        return self.cash + self.locked

    def reset(self):
        self.cash = self.peak = self.initial
        self.pending = []
        self.i = self.bets = 0
        self.staked = self.max_drawdown = 0.
        self.done = False
        self.floor_hit = False
        self.curve = [self.initial]
        self._observe_peak()
        return self.observation()

    def _observe_peak(self):
        self.peak = max(self.peak, self.equity)
        self.max_drawdown = max(self.max_drawdown, 1 - self.equity / self.peak)
        self.floor_hit |= self.equity <= self.initial * self.floor

    def _settle(self, until=None):
        due = sorted([x for x in self.pending if until is None or x[0] <= until], key=lambda x: x[0])
        # Settle equal timestamps together, avoiding arbitrary intra-batch drawdown.
        for time in sorted({x[0] for x in due}):
            batch = [x for x in due if x[0] == time]
            self.pending = [x for x in self.pending if x[0] != time]
            self.cash += sum(x[2] for x in batch)
            self._observe_peak()
            self.curve.append(self.equity)

    def probability(self):
        g = self.games[self.i]
        return ((1 - self.elo_weight) * g.market_home_probability +
                self.elo_weight * self.elo_probs.get(g.game_id, .5))

    def observation(self):
        if self.done:
            return np.zeros(OBS_SIZE, dtype=np.float32)
        g = self.games[self.i]
        return np.array([self.cash / self.initial, self.equity / self.initial,
            self.peak / self.initial, self.locked / self.initial,
            1 - self.equity / self.peak, self.i / len(self.games),
            1 / g.home_odds, 1 / g.away_odds,
            g.market_home_probability, self.elo_probs.get(g.game_id, .5), self.probability()], dtype=np.float32)

    def action_mask(self):
        if self.done:
            return np.array([True] + [False] * (len(ACTIONS)-1))
        equity = self.equity
        return np.array([side is None or
            (f * equity <= self.cash + 1e-9 and
             self.locked + f * equity <= self.exposure_limit * equity + 1e-9)
            for side, f in ACTIONS])

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode has ended')
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS) or not self.action_mask()[action]:
            raise ValueError('Invalid or unaffordable action')
        before, old_dd = self.equity, self.max_drawdown
        game = self.games[self.i]
        side, fraction = ACTIONS[action]
        if side:
            stake = fraction * self.equity
            odds = game.home_odds if side == 'home' else game.away_odds
            won = game.home_win if side == 'home' else not game.home_win
            self.cash -= stake
            self.pending.append((game.settlement_time, stake, stake * odds if won else 0.))
            self.bets += 1
            self.staked += stake
        self.i += 1
        self._settle(None if self.i == len(self.games) else self.games[self.i].decision_time)
        if self.i == len(self.games) or self.floor_hit:
            self._settle()  # Stop taking bets, but honor all outstanding commitments.
            self.done = True
        reward = math.log(max(self.equity, 1e-12) / before)
        reward -= self.risk_penalty * (self.max_drawdown - old_dd)
        return self.observation(), reward, self.done, self.metrics()

    def metrics(self):
        return dict(final_bankroll=self.equity, log_growth=math.log(max(self.equity, 1e-12)/self.initial),
            max_drawdown=self.max_drawdown, floor_hit=self.floor_hit, bets=self.bets,
            total_staked=self.staked, roi_on_stakes=(self.equity-self.initial)/self.staked if self.staked else 0.)
