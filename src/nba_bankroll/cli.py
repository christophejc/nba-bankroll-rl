import argparse
import copy
import csv
import json
import pickle
import sys
import hashlib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import numpy as np
from .agents import DQN, TabularQ, baseline_action
from .data import load_games, write_demo
from .env import BankrollEnv
from .features import elo_probabilities


def run(env, policy, epsilon=0., learn=False):
    obs = env.reset()
    total = 0.
    while not env.done:
        action = baseline_action(env, policy) if isinstance(policy, str) else policy.act(obs, env.action_mask(), epsilon)
        nxt, reward, done, _ = env.step(action)
        if learn:
            policy.learn(obs, action, reward, nxt, done, env.action_mask())
        obs = nxt
        total += reward
    return {**env.metrics(), 'reward': total}, env.curve


def main():
    parser = argparse.ArgumentParser(description='NBA bankroll research CLI (simulated funds only)')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo-data', help='Create clearly synthetic demonstration CSV')
    demo.add_argument('--out', default='data/processed/synthetic.csv')
    demo.add_argument('--seed', type=int, default=7)
    imp = commands.add_parser('import-odds', help='Merge downloaded decimal-odds snapshots with results')
    imp.add_argument('--snapshots', nargs='+', required=True)
    imp.add_argument('--results', required=True)
    imp.add_argument('--bookmaker', required=True)
    imp.add_argument('--out', required=True)
    archive = commands.add_parser('build-nba-data', help='Build audited real NBA closing-price benchmark')
    archive.add_argument('--raw-dir', default='data/raw/espn_archive')
    archive.add_argument('--out', default='data/processed')
    check = commands.add_parser('validate-data')
    check.add_argument('--data', required=True)
    for command in ['backtest', 'train']:
        p = commands.add_parser(command)
        p.add_argument('--data', required=True)
        p.add_argument('--out', default=f'outputs/{command}')
        p.add_argument('--risk-penalty', type=float, default=1.)
        p.add_argument('--elo-weight', type=float, default=0., help='Fixed blend: 0=market only, 1=Elo only')
        p.add_argument('--seed', type=int, default=7)
        if command == 'train':
            p.add_argument('--algorithm', choices=['qlearning', 'dqn'], default='qlearning')
            p.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    if args.command == 'demo-data':
        write_demo(args.out, seed=args.seed)
        print(f'Wrote SYNTHETIC smoke-test data: {args.out}')
        return
    if args.command == 'import-odds':
        from .importers import import_snapshots
        print(json.dumps(import_snapshots(args.snapshots, args.results, args.bookmaker, args.out), indent=2))
        return
    if args.command == 'build-nba-data':
        from .archive import build_archive
        report = build_archive(args.raw_dir, args.out)
        print(json.dumps({k: report[k] for k in ['dataset', 'games', 'seasons', 'excluded_games', 'benchmark_only']}, indent=2))
        return
    games = load_games(args.data)
    seasons = sorted({g.season for g in games}, key=lambda s: min(g.decision_time for g in games if g.season == s))
    if args.command == 'validate-data':
        print(json.dumps({'games': len(games), 'seasons': seasons, 'bookmakers': sorted({g.bookmaker for g in games})}, indent=2))
        return
    if len(seasons) < 3:
        parser.error('Need at least three seasons for chronological train/validation/test separation')
    # Reject interleaved season labels that would invalidate this split.
    for left, right in zip(seasons, seasons[1:]):
        if max(g.settlement_time for g in games if g.season == left) > min(g.decision_time for g in games if g.season == right):
            parser.error('Season intervals overlap; cannot construct a clean chronological split')
    elo = elo_probabilities(games)
    def environment(season):
        return BankrollEnv([g for g in games if g.season == season],
            risk_penalty=args.risk_penalty, elo_probs=elo, elo_weight=args.elo_weight)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ['numpy', 'torch']:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    with open(args.data, newline='') as f:
        data_rows = list(csv.DictReader(f))
    benchmark_only = any(str(r.get('benchmark_only', '')).lower() == 'true' for r in data_rows)
    if benchmark_only:
        print('BENCHMARK ONLY: quote timestamps unavailable; decision/settlement times are proxies.')
    metadata = {**vars(args), 'benchmark_only': benchmark_only, 'python': sys.version, 'versions': versions,
        'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(), 'split': {'train': seasons[:-2], 'validation': seasons[-2:-1], 'test': seasons[-1:]},
        'synthetic': any(g.bookmaker == 'SYNTHETIC' for g in games),
        'note': 'Historical replay; outstanding bets valued at cost. No guaranteed betting edge.'}
    (out/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    results, curves = [], {}
    if args.command == 'train':
        if args.epochs < 1:
            parser.error('--epochs must be positive')
        agent = TabularQ(args.seed) if args.algorithm == 'qlearning' else DQN(args.seed)
        rng = np.random.default_rng(args.seed)
        best_score, best, history = -float('inf'), None, []
        for epoch in range(args.epochs):
            epsilon = max(.05, 1 - epoch / max(1, args.epochs * .8))
            for season in rng.permutation(seasons[:-2]):
                run(environment(season), agent, epsilon, learn=True)
            validation, _ = run(environment(seasons[-2]), agent)
            score = validation['reward']
            history.append({'epoch': epoch+1, 'epsilon': epsilon, 'validation_reward': score})
            if score > best_score:
                best_score = score
                # Preserve selected parameters; replay/optimizer state is not needed for evaluation.
                best = copy.deepcopy(agent.table if isinstance(agent, TabularQ) else agent.online.state_dict())
        if isinstance(agent, TabularQ):
            agent.table = best
            with (out/'qlearning.pkl').open('wb') as f:
                pickle.dump(best, f)
        else:
            agent.online.load_state_dict(best)
            agent.torch.save(best, out/'dqn.pt')
        (out/'training.json').write_text(json.dumps(history, indent=2)+'\n')
        metrics, curve = run(environment(seasons[-1]), agent)
        results.append({'policy': args.algorithm, 'season': seasons[-1], **metrics})
        curves[args.algorithm] = curve
    for name in ['skip', 'fixed', 'half-kelly']:
        metrics, curve = run(environment(seasons[-1]), name)
        results.append({'policy': name, 'season': seasons[-1], **metrics})
        curves[name] = curve
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out/'bankroll_curves.json').write_text(json.dumps(curves, indent=2)+'\n')
    print(json.dumps(results, indent=2))
    print(f'Artifacts: {out.resolve()}')


if __name__ == '__main__':
    main()
