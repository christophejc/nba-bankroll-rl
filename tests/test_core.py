import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from nba_bankroll.data import Game, FIELDS, load_games, write_demo
from nba_bankroll.env import BankrollEnv
from nba_bankroll.features import elo_probabilities
from nba_bankroll.agents import baseline_action, TabularQ
from nba_bankroll.cli import run
from nba_bankroll.importers import import_snapshots

T = datetime(2024, 1, 1, tzinfo=timezone.utc)


def game(i=0, win=True, decision=None, settlement=None):
    decision = decision or T+timedelta(hours=i)
    return Game(str(i), '2023-24', 'A', 'B', decision,
        decision+timedelta(minutes=30), settlement or decision+timedelta(hours=3),
        2., 2., win, 'test')


class CoreTests(unittest.TestCase):
    def test_win_and_loss_payout(self):
        for won, expected in [(True, 1050.), (False, 950.)]:
            env = BankrollEnv([game(win=won)])
            env.reset()
            env.step(3)  # Home, 5%.
            self.assertAlmostEqual(env.cash, expected)
            self.assertEqual(env.locked, 0)

    def test_away_action(self):
        env = BankrollEnv([game(win=False)])
        env.reset()
        env.step(6)
        self.assertAlmostEqual(env.cash, 1050.)

    def test_overlapping_games_lock_stakes(self):
        env = BankrollEnv([game(i) for i in range(3)])
        env.reset()
        _, reward, done, _ = env.step(3)
        self.assertEqual(env.cash, 950.)
        self.assertEqual(env.locked, 50.)
        self.assertEqual(env.equity, 1000.)
        self.assertEqual(reward, 0.)
        self.assertFalse(done)

    def test_exposure_mask_and_rejection(self):
        env = BankrollEnv([game(i) for i in range(3)], exposure_limit=.05)
        env.reset()
        env.step(3)
        self.assertEqual(env.action_mask().tolist(), [True]+[False]*6)
        with self.assertRaises(ValueError):
            env.step(3)

    def test_reward_telescopes(self):
        env = BankrollEnv([game(i, win=i % 2 == 0) for i in range(8)], risk_penalty=2.)
        metrics, _ = run(env, TabularQ())
        self.assertAlmostEqual(metrics['reward'], metrics['log_growth']-2*metrics['max_drawdown'])
        env.reset()
        reward = 0.
        while not env.done:
            _, r, _, _ = env.step(3 if env.action_mask()[3] else 0)
            reward += r
        self.assertAlmostEqual(reward, np.log(env.equity/env.initial)-2*env.max_drawdown)

    def test_market_only_with_vig_skips(self):
        env = BankrollEnv([replace(game(), home_odds=1.9, away_odds=1.9)])
        env.reset()
        self.assertEqual(baseline_action(env, 'half-kelly'), 0)
        self.assertEqual(baseline_action(env, 'fixed'), 0)

    def test_elo_waits_for_settlement(self):
        games = [game(0), game(1), game(4)]
        probabilities = elo_probabilities(games)
        self.assertEqual(probabilities['0'], probabilities['1'])
        self.assertGreater(probabilities['4'], probabilities['1'])
        flipped = [replace(games[0], home_win=False), games[1], games[2]]
        other = elo_probabilities(flipped)
        self.assertEqual(other['1'], probabilities['1'])
        self.assertNotEqual(other['4'], probabilities['4'])

    def test_floor_stops_new_bets_and_settles_all(self):
        env = BankrollEnv([game(i, win=False) for i in range(20)], bankroll_floor=.99)
        env.reset()
        while not env.done:
            env.step(3 if env.action_mask()[3] else 0)
        self.assertTrue(env.floor_hit)
        self.assertLess(env.i, 20)
        self.assertEqual(env.locked, 0.)

    def test_demo_roundtrip_and_invalid_odds(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'demo.csv'
            write_demo(path, seasons=3, games_per_season=10)
            self.assertEqual(len(load_games(path)), 30)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            rows[0]['home_odds'] = 'nan'
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader(); w.writerows(rows)
            with self.assertRaises(ValueError):
                load_games(path)

    def test_importer_latest_pregame_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d/'results.csv').write_text('game_id,season,settlement_time,home_win\nx,2023-24,2024-01-02T03:00:00Z,1\n')
            event = dict(id='x', commence_time='2024-01-02T00:00:00Z', home_team='A', away_team='B',
                bookmakers=[dict(key='book', last_update='2024-01-01T20:00:00Z',
                    markets=[dict(key='h2h', outcomes=[dict(name='A', price=1.9), dict(name='B', price=2.)])])])
            paths = []
            for hour in [21, 22]:
                p = d/f'{hour}.json'; paths.append(p)
                p.write_text(json.dumps(dict(timestamp=f'2024-01-01T{hour}:00:00Z', data=[event])))
            stats = import_snapshots(paths, d/'results.csv', 'book', d/'games.csv')
            self.assertEqual(stats['imported'], 1)
            self.assertEqual(load_games(d/'games.csv')[0].decision_time.hour, 22)


if __name__ == '__main__':
    unittest.main()
