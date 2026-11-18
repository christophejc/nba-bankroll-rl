import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from nba_bankroll.archive import american_to_decimal, build_archive


class ArchiveTests(unittest.TestCase):
    def fixture(self, root, bad_score=False):
        def zipped(name, rows):
            with gzip.open(root/name, 'wt', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
        team = dict(game_id='g', is_home='1', season_start='2023', season_type='Regular Season',
                    team_abbr='DEN', opponent_abbr='LAL', points='119', opponent_points='107')
        zipped('team_games.csv.gz', [team, {**team, 'is_home':'0', 'team_abbr':'LAL',
               'opponent_abbr':'DEN', 'points':'107', 'opponent_points':'119'}])
        game = dict(game_id='g', season_start='2023', season_type='Regular Season',
            source_event_id='e', home_abbr='DEN', away_abbr='LAL', home_espn_id='7', away_espn_id='13',
            home_score='119', away_score='107', join_status='verified', score_verified='True')
        zipped('odds_event_crosswalk.csv.gz', [game])
        quote = dict(game_id='g', season_type='Regular Season', snapshot='close', provider_id='100',
            bookmaker='DraftKings', home_moneyline='-200', away_moneyline='170', quote_quality='ok',
            benchmark_eligible='True', score_verified='True', source_event_id='e', home_abbr='DEN',
            away_abbr='LAL', scheduled_start_utc='2023-10-24T23:30:00Z', source_url='https://example.com')
        # ESPN BET has a better home price; fixed precedence must still choose DraftKings.
        zipped('odds_moneyline.csv.gz', [quote, {**quote, 'provider_id':'58', 'bookmaker':'ESPN BET',
                                                 'home_moneyline':'-180'}])
        event = dict(id='e', competitions=[dict(date='2023-10-24T23:30:00Z',
            status={'type':{'completed':True}}, competitors=[
                dict(homeAway='home', id='7', score='118' if bad_score else '119'),
                dict(homeAway='away', id='13', score='107')])])
        (root/'espn_scoreboard_202310.json').write_text(json.dumps({'events':[event]}))

    def test_moneyline_conversion_and_invalid_values(self):
        self.assertEqual(american_to_decimal(-200), 1.5)
        self.assertEqual(american_to_decimal(170), 2.7)
        for value in [0, 99, -99, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                american_to_decimal(value)

    def test_source_precedence_and_explicit_proxy_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            report = build_archive(root, root/'out')
            with (root/'out/nba_closing_benchmark.csv').open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row['bookmaker'], 'DraftKings')
            self.assertEqual(float(row['home_odds']), 1.5)
            self.assertEqual(row['quote_timestamp_utc'], '')
            self.assertEqual(row['benchmark_only'], 'True')
            self.assertEqual(report['independently_score_verified_games'], 1)
            self.assertIn('2023-10-25T16:00:00', row['settlement_time'])

    def test_score_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root, bad_score=True)
            with self.assertRaisesRegex(ValueError, 'disagree across sources'):
                build_archive(root, root/'out')
            self.assertFalse((root/'out/nba_closing_benchmark.csv').exists())

    def test_missing_primary_provider_falls_back_without_best_price_search(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            p = root/'odds_moneyline.csv.gz'
            with gzip.open(p,'rt') as f:
                rows = list(csv.DictReader(f))
            with gzip.open(p,'wt',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerow(rows[1])
            report = build_archive(root, root/'out')
            self.assertEqual(report['seasons']['2023-24']['bookmakers'], {'ESPN BET':1})


if __name__ == '__main__':
    unittest.main()
