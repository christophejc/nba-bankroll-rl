"""Validated canonical data and explicitly synthetic smoke-test fixtures."""
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

FIELDS = ['game_id', 'season', 'home_team', 'away_team', 'decision_time',
          'start_time', 'settlement_time', 'home_odds', 'away_odds',
          'home_win', 'bookmaker']


def timestamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamps must include a timezone')
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class Game:
    game_id: str
    season: str
    home_team: str
    away_team: str
    decision_time: datetime
    start_time: datetime
    settlement_time: datetime
    home_odds: float
    away_odds: float
    home_win: bool
    bookmaker: str

    @property
    def market_home_probability(self):
        # Remove the two-way overround proportionally. This is a feature,
        # not an assertion that the market probabilities are ground truth.
        h, a = 1 / self.home_odds, 1 / self.away_odds
        return h / (h + a)


def load_games(path):
    games, seen = [], set()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        for row in reader:
            if row['game_id'] in seen:
                raise ValueError(f"Duplicate game_id: {row['game_id']}")
            seen.add(row['game_id'])
            if row['home_win'] not in {'0', '1'}:
                raise ValueError('home_win must be 0 or 1; exclude ties/void games')
            game = Game(**{**{k: row[k] for k in FIELDS},
                **{k: timestamp(row[k]) for k in ['decision_time', 'start_time', 'settlement_time']},
                'home_odds': float(row['home_odds']), 'away_odds': float(row['away_odds']),
                'home_win': row['home_win'] == '1'})
            if not game.decision_time < game.start_time < game.settlement_time:
                raise ValueError('Require decision_time < start_time < settlement_time')
            if not all(math.isfinite(o) and o > 1 for o in (game.home_odds, game.away_odds)):
                raise ValueError('Decimal odds must be finite and > 1')
            if not all(str(row[k]).strip() for k in FIELDS):
                raise ValueError('Empty fields are not allowed')
            games.append(game)
    if not games:
        raise ValueError('Dataset is empty')
    return sorted(games, key=lambda g: (g.decision_time, g.game_id))


def write_demo(path, seasons=6, games_per_season=240, seed=7):
    """Fabricated teams, prices, outcomes and dates; never empirical evidence."""
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    teams = [f'Synthetic Team {i:02}' for i in range(12)]
    strength = {t: rng.gauss(0, .6) for t in teams}
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for year in range(2018, 2018 + seasons):
            for i in range(games_per_season):
                h, a = rng.sample(teams, 2)
                p = 1 / (1 + math.exp(-(strength[h] - strength[a] + .2)))
                market = min(.92, max(.08, p + rng.gauss(0, .04)))
                start = datetime(year, 10, 1, 23, tzinfo=timezone.utc) + timedelta(days=i // 4, minutes=30 * (i % 4))
                writer.writerow(dict(game_id=f'demo-{year}-{i}', season=f'{year}-{year+1}',
                    home_team=h, away_team=a, decision_time=(start-timedelta(hours=1)).isoformat(),
                    start_time=start.isoformat(), settlement_time=(start+timedelta(hours=3)).isoformat(),
                    home_odds=round(1 / (market * 1.025), 4),
                    away_odds=round(1 / ((1-market) * 1.025), 4),
                    home_win=int(rng.random() < p), bookmaker='SYNTHETIC'))
