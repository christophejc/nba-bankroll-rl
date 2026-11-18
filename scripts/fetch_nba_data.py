"""Fetch pinned public data and optional independent ESPN scoreboard checks.

Downloads data only. Never imports or runs upstream repository code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'johnson-lee-v0/NBA-Games-Outcome'
COMMIT = '4e6d2ab76302339d1d6880cce217713877eec3d8'
FILES = {
    'data/processed/odds_moneyline.csv.gz': '78c710c6897dd137ffe947b2fc314e2697e34d885964ee9f9159e69d08929572',
    'data/processed/odds_event_crosswalk.csv.gz': '1dfce11243063afd7103399e6f0cbfcafa7ed5b47bf48011c3d5b37c0643e379',
    'data/processed/team_games.csv.gz': '9679c086899a89b3c3006cfa930a224c86cc6dbf84e22a839fe681c55b4204a7',
    'data/odds_source_manifest.json': 'e280be1db01aa055eafec644908c9c46e54e49419f3f1161985ea74abec97eeb',
    'docs/odds-sources.md': '46b0ba21683359291bc40616fbea1b291cf7f915a39081400dfcd167e1c20e95',
    'DATA_SOURCES.md': '85d553f9f63632a4eae65059dcbece1d8cec20c23938375ae963e87c19443fa6',
    'LICENSE': '308974c784ee992ea2049eeb8c83f5227b014942cd10c27573094b859956f1d8',
}


def fetch(url, path, expected=None):
    if path.exists():
        data = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={'User-Agent': 'nba-bankroll-academic-data-check/0.1'})
        data = urllib.request.urlopen(request, timeout=60).read()
        if not expected or hashlib.sha256(data).hexdigest() == expected:
            path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return dict(file=str(path), url=url, sha256=digest, bytes=len(data))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='data/raw/espn_archive')
    p.add_argument('--verify-scoreboards', action='store_true', help='Fetch 27 monthly ESPN scoreboards, 2023-24 through 2025-26')
    args = p.parse_args()
    root = Path(args.raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}'
        records.append(fetch(url, root/Path(name).name, expected))
    if args.verify_scoreboards:
        months = [f'{year}{month:02}' for start in (2023, 2024, 2025)
                  for year, month in [(start, 10), (start, 11), (start, 12)] +
                  [(start+1, month) for month in range(1, 7)]]
        def scoreboard(month):
            return fetch(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={month}&limit=1000',
                         root/f'espn_scoreboard_{month}.json')
        with ThreadPoolExecutor(max_workers=3) as pool:
            records.extend(pool.map(scoreboard, months))
    from datetime import datetime, timezone
    manifest = dict(repository=REPO, commit=COMMIT, checked_at_utc=datetime.now(timezone.utc).isoformat(), files=records)
    (root/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Checked {len(records)} data/provenance files in {root}')


if __name__ == '__main__':
    main()
