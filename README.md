# NBA bankroll reinforcement learning

A Python research starter for **simulated, season-long NBA moneyline betting**.
The main question is whether learned stake allocation improves the tradeoff
between growth and maximum drawdown relative to fixed staking and fractional Kelly.
No live betting, sportsbook credentials, or real-money execution is included.

## Setup

Use Python 3.11 or newer (3.11/3.12 recommended for PyTorch compatibility):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Optional neural-network agent:
python -m pip install -e '.[rl]'
```

The baseline and tabular-Q pipeline needs only NumPy. PyTorch is optional.

## Real NBA data is loaded

See [the data audit and provenance notes](docs/nba-data.md). The local benchmark has
3,689 regular-season games across all 30 teams: 2023-24 (1,230), 2024-25 (1,230),
and 2025-26 (1,229). One game has no eligible closing price. Older team-game logs
are also loaded separately for future lagged context features.

```sh
nba-bankroll validate-data --data data/processed/nba_closing_benchmark.csv
nba-bankroll backtest --data data/processed/nba_closing_benchmark.csv --out outputs/nba-market-baselines
```

These are real results and prices, but **quote-publication timestamps are missing**.
Decision and settlement timestamps are labeled simulation assumptions. Use this as
a closing-price benchmark; exact execution claims require timestamped prices.
Reproduce the download and build with:

```sh
python scripts/fetch_nba_data.py --verify-scoreboards
nba-bankroll build-nba-data
```

## Run a complete smoke test

```sh
nba-bankroll demo-data
nba-bankroll validate-data --data data/processed/synthetic.csv
nba-bankroll backtest --data data/processed/synthetic.csv
nba-bankroll train --data data/processed/synthetic.csv --algorithm qlearning --epochs 30
nba-bankroll train --data data/processed/synthetic.csv --algorithm dqn --epochs 30 --out outputs/dqn
python -m unittest discover -s tests -v
```

**Demo data is fabricated, including its teams, odds, dates, and outcomes. It is
only for checking code. It cannot establish that a strategy works on NBA games.**
`--elo-weight 0.2` optionally blends 20% chronological Elo with 80% market probability.
The default is zero: market probabilities alone, with no asserted predictive edge.
On ordinary positive-margin prices, the value-betting baselines should skip.
The Elo blend is a transparent experimental feature, not a calibrated predictor.
Select it on validation data, never by inspecting test returns.

Each run writes `config.json`, `metrics.csv`, and `bankroll_curves.json`.
Training also writes validation history and the validation-selected checkpoint.
Separate `--out` directories for each configuration and seed to retain experiments.

## Real-data contract

One row per game, one bookmaker and one decision snapshot per game. All dates must
include timezone offsets. Prices must be **decimal**, not American moneylines.
Required CSV columns:

```csv
game_id,season,home_team,away_team,decision_time,start_time,settlement_time,home_odds,away_odds,home_win,bookmaker
example,2023-24,Home Team,Away Team,2024-01-01T23:00:00Z,2024-01-02T00:00:00Z,2024-01-02T03:00:00Z,1.80,2.10,1,example-book
```

`home_win` is 0 or 1; settlement must follow start, which must follow decision.
Include only completed, nonvoid, two-way regular-season NBA games, with overtime
included in the result. Exclude canceled games and the NBA Cup final if your
chosen regular-season definition excludes it. Deduplicate provider event IDs.
Use recorded completion/settlement timestamps where available. If estimating
settlement times, document the approximation and test conservative delays.
Do not include a game's final stats as its pregame features.

Raw historical odds are not committed. The free local benchmark above is already
loaded; for timestamped data, a practical alternative is The Odds API's historical
endpoint (paid historical access):
https://the-odds-api.com/historical-odds-data/
Request `oddsFormat=decimal`, the NBA sport, and `h2h`. The offline adapter accepts
its historical wrapper `{timestamp, data: [...]}` JSON. Download the chosen
snapshots yourself; no API keys need to enter this repository.

Prepare a separate results CSV joined by provider event ID:

```csv
game_id,season,settlement_time,home_win
provider-event-id,2023-24,2024-01-02T03:00:00Z,1
```

```sh
nba-bankroll import-odds --snapshots data/raw/snapshot-*.json \
  --results data/raw/results.csv --bookmaker draftkings \
  --out data/processed/nba.csv
nba-bankroll validate-data --data data/processed/nba.csv
nba-bankroll train --data data/processed/nba.csv --algorithm dqn --out outputs/nba-dqn
```

The adapter chooses the latest supplied pregame snapshot per game and reports
unmatched result rows. Supply snapshots for your intended decision horizon;
feeding every snapshot creates a near-start experiment, not a day-ahead one.
Audit coverage, price staleness, missing games and result joins before research.
The adapter validates schema locally; it has not been tested against a paid feed.
Market-average prices need to be labeled as a simulated benchmark, not executable
bookmaker prices. Do not distribute raw licensed provider data in GitHub.

## Environment and methods

- A season is an episode, starting with 1,000 simulated units.
- Actions: skip; home or away at 1%, 2%, or 5% of current equity.
- Outstanding stakes are locked. Equity is cash plus stakes valued at cost.
  Future payouts are never exposed as observations.
- Total outstanding exposure is capped at 20% of current equity when placing bets.
  A loss can temporarily push existing commitments above the cap; no forced sales.
- Settlement occurs before the next decision using available timestamps. Equal-time
  settlements form one batch. Simultaneous decisions share cash and have a stable
  game-ID order; that arbitrary allocation order is a limitation to test later.
- At a 20%-of-initial-bankroll floor, stop placing bets and settle commitments.
- Observations: cash, equity, previous peak, outstanding exposure, current drawdown,
  season progress, inverse prices, de-vigged market probability, chronological Elo,
  and a fixed market/Elo probability blend. Monetary values normalize by initial cash.
- Market probability is `(1/home_odds) / (1/home_odds + 1/away_odds)`.
  Odds can summarize injuries and other information, but do not identify their cause
  or guarantee that the market incorporated every relevant update.
- Elo uses only results settled by each decision timestamp. It carries through seasons;
  no test outcomes are used before they would actually become observable.
- Reward is log equity change minus `lambda` times an increase in **maximum** drawdown.
  With undiscounted returns, this sums to final log growth minus lambda times max drawdown.
- Baselines: skip, 2% positive-estimated-value staking, and capped half Kelly. Kelly is
  rounded down to the same discrete stake grid as the agents (below 1% becomes skip).
- Tabular Q-learning uses coarse bins. DQN uses two 64-unit hidden layers, replay,
  a target network, Huber loss, gradient clipping, epsilon-greedy exploration and
  Double-DQN targets. Both use gamma=1 for the finite-season objective.

Outstanding positions are summarized by total exposure rather than individual
settlement schedules; observations are therefore a partial-state approximation,
not a claim of an exact Markov representation. A future extension can encode
pending positions and upcoming opportunities more fully.

## Evaluation protocol

Seasons are ordered by their decision dates: all but the latest two train the
agent, the penultimate selects checkpoints, and the last is held out for testing.
At least three nonoverlapping seasons are required. No decisions are shuffled
within a season; only training-season order changes. Evaluation uses greedy
policies without learning. Elo still updates from already settled results as a
normal pregame feature; policy weights stay frozen.

Compare identical data, features, action constraints and seeds. Run risk penalties
0 and 1 as the initial ablation, using separate output directories. Tune on the
validation season. Report test bankroll/log growth, max drawdown, floor hits,
turnover, ROI on stakes and all failed runs. Repeat across training seeds; these
measure training variability, not additional independent NBA seasons. Extend to
rolling temporal folds before making generalization claims.

Historical replay assumes small simulated bets do not change odds or results.
It can calculate payouts for all permitted actions without logged bettor actions;
CQL/BCQ is not necessary for this initial setup. Kelly is an important benchmark:
under suitable independent-opportunity log-growth assumptions, a complex RL
method should not be expected to beat correctly specified Kelly sizing.

## Next research steps

1. Secure a permitted historical odds/results source and audit a few sample games.
2. Establish real-data market-only, fixed-stake and half-Kelly results.
3. Build/calibrate a win model with walk-forward predictions; keep every staking
   strategy on identical probability estimates. Do not use in-sample fitted
   predictions as if they were realistic training-time forecasts.
4. Compare Q-learning, DQN and drawdown-aware DQN across temporal folds and seeds.
5. Add charts, richer team features and optional continuous-action actor-critic/PPO
   after the baseline experiment is reproducible.

## Layout

`data.py`: schema validation and synthetic generator; `features.py`: chronological
Elo; `importers.py`: offline odds adapter; `env.py`: settlement and reward;
`agents.py`: baselines/Q-learning/DQN; `cli.py`: reproducible experiment entry point;
`tests/`: accounting, leakage, risk and import checks.
