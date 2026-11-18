"""Build an audited closing-price benchmark from pinned NBA/ESPN archives.

No exact historical quote or settlement times are asserted. Proxy timestamps
support a conservative replay; metadata travels with every output row.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, time, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
from .data import FIELDS, load_games, timestamp

PROVIDER_ORDER = ['100', '40', '58']  # DraftKings, historical DraftKings, ESPN BET.
EXTRA = ['source_event_id', 'home_score', 'away_score', 'home_moneyline', 'away_moneyline',
         'provider_id', 'snapshot', 'quote_timestamp_utc', 'timestamp_precision',
         'benchmark_only', 'decision_time_assumption', 'settlement_time_assumption', 'source_url']


def american_to_decimal(value):
    value = float(value)
    if not math.isfinite(value) or abs(value) < 100:
        raise ValueError('American moneyline must be finite and <= -100 or >= 100')
    return 1 + (value/100 if value > 0 else 100/abs(value))


def read_gzip(path):
    with gzip.open(path, 'rt', newline='') as f:
        return list(csv.DictReader(f))


def scoreboard_index(root):
    result = {}
    for path in sorted(Path(root).glob('espn_scoreboard_*.json')):
        for event in json.loads(path.read_text())['events']:
            comp = event['competitions'][0]
            sides = {t['homeAway']: t for t in comp['competitors']}
            if not comp['status']['type']['completed']:
                continue
            result[str(event['id'])] = dict(start=timestamp(comp['date']),
                home_id=str(sides['home']['id']), away_id=str(sides['away']['id']),
                home_score=int(sides['home']['score']), away_score=int(sides['away']['score']))
    return result


def build_archive(raw_dir, out_dir):
    root, output = Path(raw_dir), Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    quotes = read_gzip(root/'odds_moneyline.csv.gz')
    crosswalk = read_gzip(root/'odds_event_crosswalk.csv.gz')
    teams = read_gzip(root/'team_games.csv.gz')
    checks = scoreboard_index(root)
    if not checks:
        raise ValueError('Fetch independent scoreboards first: scripts/fetch_nba_data.py --verify-scoreboards')
    home_rows = {r['game_id']: r for r in teams if r['is_home'] == '1'}
    if len(home_rows)*2 != len(teams):
        raise ValueError('Team-game archive has duplicate or missing home perspectives')
    by_game = defaultdict(list)
    rejected_quotes, excluded_games, rows = [], [], []
    for q in quotes:
        if q['season_type'] != 'Regular Season' or q['snapshot'] != 'close':
            continue
        if q['provider_id'] not in PROVIDER_ORDER:
            continue
        try:
            h, a = american_to_decimal(q['home_moneyline']), american_to_decimal(q['away_moneyline'])
            margin = 1/h + 1/a - 1
            if not -.02 <= margin <= .20 or q['quote_quality'] != 'ok':
                raise ValueError('Implausible two-way overround or flagged source quality')
            if q['score_verified'] != 'True' or q['benchmark_eligible'] != 'True':
                raise ValueError('Source did not verify/accept quote')
        except ValueError as exc:
            rejected_quotes.append({'game_id': q['game_id'], 'reason': str(exc), 'provider_id': q['provider_id']})
            continue
        by_game[q['game_id']].append(q)
    for g in crosswalk:
        if g['season_type'] != 'Regular Season':
            continue
        gid = g['game_id']
        if g['join_status'] != 'verified' or g['score_verified'] != 'True':
            excluded_games.append({'game_id': gid, 'reason': 'Unverified source crosswalk'})
            continue
        direct = checks.get(g['source_event_id'])
        source = home_rows.get(gid)
        if direct is None:
            excluded_games.append({'game_id': gid, 'reason': 'Missing independent ESPN scoreboard'})
            continue
        if source is None:
            raise ValueError(f'Missing team-game record: {gid}')
        if not (g['home_espn_id'] == direct['home_id'] and g['away_espn_id'] == direct['away_id']
                and int(g['home_score']) == direct['home_score'] == int(source['points'])
                and int(g['away_score']) == direct['away_score'] == int(source['opponent_points'])
                and g['home_abbr'] == source['team_abbr'] and g['away_abbr'] == source['opponent_abbr']
                and g['season_start'] == source['season_start']
                and g['season_type'] == source['season_type']):
            raise ValueError(f'Teams, scores or season disagree across sources: {gid}')
        available = by_game.get(gid, [])
        if not available:
            excluded_games.append({'game_id': gid, 'reason': 'No eligible closing price in fixed provider order'})
            continue
        selected = min(available, key=lambda q: PROVIDER_ORDER.index(q['provider_id']))
        if sum(q['provider_id'] == selected['provider_id'] for q in available) != 1:
            raise ValueError(f'Duplicate selected quote: {gid}')
        if not (selected['source_event_id'] == g['source_event_id'] and selected['home_abbr'] == g['home_abbr']
                and selected['away_abbr'] == g['away_abbr']
                and timestamp(selected['scheduled_start_utc']) == direct['start']):
            raise ValueError(f'Quote identity/start mismatch: {gid}')
        start = direct['start']
        # Proxy decision immediately before scheduled start; quote timing is unknown.
        decision = start - timedelta(minutes=1)
        local = start.astimezone(ZoneInfo('America/New_York'))
        settle = datetime.combine(local.date()+timedelta(days=1), time(12), tzinfo=local.tzinfo).astimezone(timezone.utc)
        settle = max(settle, start+timedelta(hours=6))
        if direct['home_score'] == direct['away_score']:
            raise ValueError(f'Tied completed NBA game: {gid}')
        year = int(g['season_start'])
        rows.append(dict(game_id=gid, season=f'{year}-{str(year+1)[-2:]}',
            home_team=g['home_abbr'], away_team=g['away_abbr'],
            decision_time=decision.isoformat(), start_time=start.isoformat(), settlement_time=settle.isoformat(),
            home_odds=american_to_decimal(selected['home_moneyline']),
            away_odds=american_to_decimal(selected['away_moneyline']),
            home_win=int(direct['home_score'] > direct['away_score']), bookmaker=selected['bookmaker'],
            source_event_id=g['source_event_id'], home_score=direct['home_score'], away_score=direct['away_score'],
            home_moneyline=selected['home_moneyline'], away_moneyline=selected['away_moneyline'],
            provider_id=selected['provider_id'], snapshot='close', quote_timestamp_utc='',
            timestamp_precision='opening/closing label only; publication timestamp unavailable', benchmark_only=True,
            decision_time_assumption='scheduled_start_minus_1_minute_proxy_not_observed_quote_time',
            settlement_time_assumption='next_calendar_day_noon_America_New_York_minimum_6_hours_after_start',
            source_url=selected['source_url']))
    rows.sort(key=lambda r: (r['decision_time'], r['game_id']))
    target = output/'nba_closing_benchmark.csv'
    with target.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS+EXTRA)
        w.writeheader(); w.writerows(rows)
    load_games(target)
    # Separate retrospective outcomes/stats from the agent's pregame observations.
    stat_fields = ['game_id', 'season_start', 'season_type', 'game_date', 'team_id', 'team_abbr',
        'opponent_id', 'opponent_abbr', 'is_home', 'points', 'opponent_points', 'fga', 'fgm',
        'fg3a', 'fg3m', 'fta', 'ftm', 'turnovers', 'rebounds', 'assists', 'steals', 'won', 'data_quality']
    regular = [r for r in teams if r['season_type'] == 'Regular Season']
    with (output/'nba_team_game_logs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=stat_fields, extrasaction='ignore')
        w.writeheader(); w.writerows(regular)
    seasons = {}
    for season in sorted({r['season'] for r in rows}):
        cohort = [r for r in rows if r['season'] == season]
        team_counts = Counter(t for r in cohort for t in (r['home_team'],r['away_team']))
        seasons[season] = dict(games=len(cohort), teams=len(team_counts), team_game_counts=dict(sorted(team_counts.items())),
            bookmakers=dict(Counter(r['bookmaker'] for r in cohort)),
            first_start=cohort[0]['start_time'], last_start=cohort[-1]['start_time'])
    report = dict(dataset=str(target), games=len(rows), seasons=seasons,
        regular_season_targets=sum(g['season_type']=='Regular Season' for g in crosswalk),
        excluded_games=excluded_games, rejected_quotes=rejected_quotes,
        independently_score_verified_games=len(rows), independent_scoreboard_events=len(checks),
        provider_order=PROVIDER_ORDER, team_log_rows=len(regular),
        team_log_seasons=sorted({r['season_start'] for r in regular}),
        benchmark_only=True, quote_publication_timestamps_available=False,
        timing_assumptions={'decision':'scheduled start minus one minute, a proxy only',
            'settlement':'next day noon US/Eastern, at least six hours after scheduled start'},
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
                      if p.name.endswith('.csv.gz')},
        limitations=['Prices have no verifiable as-of publication timestamps; not an execution-grade backtest.',
            'Settlement delays are assumptions, not observed completion times.',
            'Provider varies using a fixed precedence; results need bookmaker-stratified sensitivity checks.',
            'Only three odds seasons; one validation and one test season are limited evidence.',
            'Box scores are postgame labels: shift/lag them before constructing pregame features.',
            'Older team logs are loaded separately and are not yet used by the agent.'])
    (output/'nba_data_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
