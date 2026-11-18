"""Outcome-derived features update only after a game's recorded settlement."""
import heapq


def elo_probabilities(games, k=20., home_advantage=65.):
    ratings, pending, result = {}, [], {}
    for g in sorted(games, key=lambda x: (x.decision_time, x.game_id)):
        while pending and pending[0][0] <= g.decision_time:
            _, _, old, predicted = heapq.heappop(pending)
            delta = k * (float(old.home_win) - predicted)
            ratings[old.home_team] = ratings.get(old.home_team, 1500.) + delta
            ratings[old.away_team] = ratings.get(old.away_team, 1500.) - delta
        difference = ratings.get(g.home_team, 1500.) - ratings.get(g.away_team, 1500.)
        p = 1 / (1 + 10 ** (-(difference + home_advantage) / 400))
        result[g.game_id] = p
        heapq.heappush(pending, (g.settlement_time, g.game_id, g, p))
    return result
