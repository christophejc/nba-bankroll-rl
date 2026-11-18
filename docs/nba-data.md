# Loaded NBA data

## Coverage

The local closing-moneyline benchmark includes all 30 NBA teams:

| Season | Included regular-season games | Coverage |
| --- | ---: | --- |
| 2023-24 | 1,230 | Complete |
| 2024-25 | 1,230 | Complete |
| 2025-26 | 1,229 | One game lacks an eligible closing quote |
| Total | 3,689 | Playoffs, play-ins and non-standings Cup finals excluded |

The missing game is NBA ID `0022500086`. No odds were imputed. Its result is
retained in the separate team-game log. Every included result was checked against
independently downloaded ESPN monthly scoreboards, matching home/away team IDs,
scheduled starts and both final scores. Team logs independently agree with the
crosswalk on scores and NBA team abbreviations.

Files (generated locally, ignored by Git):

- `data/processed/nba_closing_benchmark.csv`: prices, labels, source links and timing assumptions.
- `data/processed/nba_team_game_logs.csv`: regular-season team statistics, 2017-18 through 2025-26.
- `data/processed/nba_data_audit.json`: exclusions, team/season coverage, bookmaker counts and checksums.
- `data/raw/espn_archive/`: untouched pinned source files, ESPN verification responses and download manifest.

The older team logs are available for future lagged features. They are **not yet
wired into the environment**. The agent's current Elo features still use the games
in the supplied betting CSV. No upstream model predictions or fitted features are
used as inputs.

## Provenance and selection

The downloaded public archive is:
https://github.com/johnson-lee-v0/NBA-Games-Outcome

Pinned commit: `4e6d2ab76302339d1d6880cce217713877eec3d8`.
The downloader verifies fixed SHA-256 values before using its files. Only data and
source notices were downloaded; no upstream code was executed.

The archive's ESPN historical odds were checked against one original ESPN odds
response, and **all included final scores and event identities** were checked
against original monthly ESPN scoreboards. The single odds spot check is not an
independent re-fetch of every historical price.

Primary odds source pattern:
`https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event_id}/competitions/{event_id}/odds?limit=100`

Scoreboard source pattern:
`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMM&limit=1000`

Team-game data was derived upstream from the NBA play-by-play mirror
https://github.com/shufinskiy/nba_data . Original source descriptions, license and
attribution notices are retained in the raw directory. Repository code licenses
do not necessarily license underlying third-party sports data. Raw data remains
local rather than being committed or republished.

Prices always come as a paired home/away quote from the **same provider**.
Fixed provider precedence is DraftKings 100, historical DraftKings 40, then ESPN BET
58. This is decided without looking at outcomes or choosing the most favorable
price. Coverage changes by year:

| Season | DraftKings games | ESPN BET games |
| --- | ---: | ---: |
| 2023-24 | 1,229 | 1 |
| 2024-25 | 0 | 1,230 |
| 2025-26 | 968 | 261 |

Prices are converted from American to decimal odds. Paired implied probabilities
must sum to between 0.98 and 1.20; invalid or source-flagged quotes are rejected.
Pick'em prices are allowed. The source's anomalous Indiana/Denver opening quote
is never selected because this dataset uses closing quotes only.

## Timing limitations

This is a **historical closing-price benchmark**, not an execution-grade replay.
The source supplies opening/closing labels but no reliable quote-publication
timestamps. Its retrieval date is not a historical quote date.

To use the existing simulator, output timestamps explicitly assume:

- Decision: scheduled start minus one minute (a proxy, not observed price availability).
- Settlement: noon US/Eastern on the next local calendar day, at least six hours after start.

That settlement convention prevents recycling stakes into later games on the same
local date. It is conservative about cash reuse, but not a reconstruction of actual
bookmaker settlement. Long delays/exceptional games still need special auditing.
Same-time decisions retain the simulator's fixed game-ID ordering.

Every output row carries `benchmark_only=True`, a blank `quote_timestamp_utc`, and
human-readable timing-assumption columns. The experiment CLI prints the limitation
and preserves the benchmark flag in its run configuration. Do not use these data
to claim prices were available precisely one minute before tip-off. Timestamped
odds snapshots are needed to support that stronger interpretation.

## Reproduce and run

```sh
source .venv/bin/activate
python scripts/fetch_nba_data.py --verify-scoreboards
nba-bankroll build-nba-data
nba-bankroll validate-data --data data/processed/nba_closing_benchmark.csv
nba-bankroll backtest --data data/processed/nba_closing_benchmark.csv \
  --out outputs/nba-market-baselines
```

Downloads use public unauthenticated endpoints and cache responses locally.
Rebuild uses local files only. No paid key or account is required for this archive.

The existing split assigns 2023-24 to training, 2024-25 to validation and 2025-26
to testing. Only one training season is currently available with these prices;
older statistics do not create additional priced betting episodes. More odds
seasons and rolling test folds would strengthen the eventual research.

The first real-data verification ran market-only baseline strategies. All skipped
every game in the test season, as expected when no edge over the quoted prices
is estimated. These are data/pipeline checks, not evidence of a profitable policy.
RL has not been trained on the real dataset as part of the ingestion task.
