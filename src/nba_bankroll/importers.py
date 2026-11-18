"""Offline adapter for The Odds API historical snapshot JSON plus settled labels."""
import csv
import json
from pathlib import Path
from .data import FIELDS, load_games, timestamp


def import_snapshots(paths, results_path, bookmaker, output):
    """Select latest supplied pregame snapshot per labeled event for one bookmaker.

    Supply snapshots at the intended decision horizon. Passing every snapshot and
    selecting the latest yields a near-start experiment, not a day-ahead one.
    Results require game_id, season, settlement_time, home_win.
    """
    with open(results_path, newline='') as f:
        labels = list(csv.DictReader(f))
    required = {'game_id', 'season', 'settlement_time', 'home_win'}
    if not labels or not required <= labels[0].keys():
        raise ValueError(f'Results CSV needs {sorted(required)}')
    results = {r['game_id']: r for r in labels}
    if len(results) != len(labels):
        raise ValueError('Duplicate result game_id')
    selected = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        decision = timestamp(payload['timestamp'])
        for event in payload['data']:
            if event.get('sport_key', 'basketball_nba') != 'basketball_nba':
                raise ValueError('Only NBA events are supported')
            if event['id'] not in results or decision >= timestamp(event['commence_time']):
                continue
            for book in event['bookmakers']:
                if book['key'] != bookmaker:
                    continue
                if timestamp(book['last_update']) > decision:
                    raise ValueError('Bookmaker update is after snapshot timestamp')
                for market in book['markets']:
                    if market['key'] != 'h2h':
                        continue
                    if market.get('last_update') and timestamp(market['last_update']) > decision:
                        raise ValueError('Market update is after snapshot timestamp')
                    prices = {o['name']: o['price'] for o in market['outcomes']}
                    home, away = event['home_team'], event['away_team']
                    if set(prices) != {home, away}:
                        raise ValueError('Expected two-way NBA moneyline market')
                    row = dict(game_id=event['id'], season=results[event['id']]['season'],
                        home_team=home, away_team=away, decision_time=decision.isoformat(),
                        start_time=event['commence_time'],
                        settlement_time=results[event['id']]['settlement_time'],
                        home_odds=prices[home], away_odds=prices[away],
                        home_win=results[event['id']]['home_win'], bookmaker=bookmaker)
                    old = selected.get(event['id'])
                    if old is None or decision > timestamp(old['decision_time']):
                        selected[event['id']] = row
    if not selected:
        raise ValueError('No labeled pregame prices matched; request decimal odds and check event IDs')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate in a temporary file before replacing the requested destination.
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', newline='', suffix='.csv', dir=output.parent, delete=False) as f:
        temp = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected.values())
    try:
        load_games(temp)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return {'imported': len(selected), 'unmatched_results': len(results)-len(selected)}
