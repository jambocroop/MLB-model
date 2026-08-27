"""
Squad Builder: assemble groups of players who must ALL clear a threshold.
============================================================================

For "pick a group of N, all must hit" style formats (e.g. Underdog/PrizePicks-
style pick'ems) -- different from single best picks or single-game value bets
against a specific book's odds. Here, odds mostly don't matter; what matters
is finding groups of players most likely to ALL clear their line together.

METHODOLOGY:
    1. Convert every batter's point projection (expected_combined or
       expected_total_bases) into P(actual > line) using the same Poisson
       approximation already validated in odds_value_finder.py -- imported
       directly from there rather than reimplemented, so both tools use
       identical, already-tested math.
    2. Rank all batters by that probability for the chosen metric/line.
    3. For a FIXED squad size, the joint probability of all N clearing
       (assuming independence) is maximized simply by taking the N highest
       individual probabilities -- no complex optimization needed; replacing
       any squad member with a higher-probability one strictly increases the
       product.
    4. Also builds diversified alternatives (--max-per-game caps how many
       players from the same game can appear together) and additional
       non-overlapping alternative squads, since the single "best" squad
       often clusters 2+ players from the same game.

IMPORTANT CAVEAT -- read before trusting the joint probability number:
    The joint probability shown is the PRODUCT of individual probabilities,
    which assumes independence. That's almost certainly wrong for players in
    the SAME game: their outcomes are positively correlated in reality (a
    rainout, a blowout, a dominant/terrible pitching performance, bullpen
    exposure -- all move multiple players on the same team together, in the
    same direction). This means:
      - For a squad with 2+ players from the same game, the TRUE joint
        probability of all hitting together is probably somewhat HIGHER than
        the naive product on the day the game goes well, but the squad also
        has more correlated DOWNSIDE risk (one bad pitching matchup or early
        rainout can take out multiple squad members at once).
      - For a fully diversified squad (one player per game), independence is
        a much more reasonable approximation.
    Treat the joint probability as a rough way to ORDER candidate squads
    against each other, not as a precise, trustworthy number on its own.

USAGE:
    python squad_builder.py --date 2026-08-26 --metric combined --line 1.5
    python squad_builder.py --date 2026-08-26 --metric total_bases --line 2.5
    python squad_builder.py --csv mlb_report_2026-08-26.csv --metric combined --line 1.5
    python squad_builder.py --date 2026-08-26 --metric combined --line 1.5 --squad-size 6 --max-per-game 1
"""

import argparse
import csv
from datetime import datetime

from mlb_daily_analysis import analyze_date
from odds_value_finder import poisson_prob_over

METRIC_FIELD = {"combined": "expected_combined", "total_bases": "expected_total_bases"}
METRIC_LABEL = {"combined": "Hits+Runs+RBIs", "total_bases": "Total Bases"}


def load_rows(date_str, team_filter, csv_path):
    if csv_path:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for key in ("expected_combined", "expected_total_bases"):
                if r.get(key) not in (None, ""):
                    r[key] = float(r[key])
        return rows
    rows, _ = analyze_date(date_str, team_filter=team_filter, use_statcast=False, workers=8)
    return rows


def compute_probabilities(rows, metric, line):
    field = METRIC_FIELD[metric]
    scored = []
    for r in rows:
        proj = r.get(field)
        if proj is None or proj == "":
            continue
        prob = poisson_prob_over(line, float(proj))
        scored.append({
            "batter": r["batter"], "team": r["team"], "game": r.get("game", ""),
            "opp_pitcher": r.get("opp_pitcher", ""), "projection": float(proj), "prob": prob,
            "note": r.get("note", ""),
        })
    scored.sort(key=lambda x: -x["prob"])
    return scored


def joint_probability(squad):
    p = 1.0
    for player in squad:
        p *= player["prob"]
    return p


def build_squad(candidates, squad_size, max_per_game=None, exclude_batters=None):
    """Greedily take the highest-probability remaining candidates, respecting
    the per-game cap and any exclusions (used for building non-overlapping
    alternative squads)."""
    exclude_batters = exclude_batters or set()
    game_counts = {}
    squad = []
    for c in candidates:
        if c["batter"] in exclude_batters:
            continue
        game = c.get("game", "")
        if max_per_game and game_counts.get(game, 0) >= max_per_game:
            continue
        squad.append(c)
        game_counts[game] = game_counts.get(game, 0) + 1
        if len(squad) == squad_size:
            break
    return squad


def print_squad(squad, label, metric_label, line, squad_size):
    if len(squad) < squad_size:
        print(f"\n{label}: only {len(squad)}/{squad_size} candidates available -- not enough eligible players.")
        return
    jp = joint_probability(squad)
    games_used = len(set(p["game"] for p in squad))
    diversity_note = "" if games_used == len(squad) else f"  (\u26a0 only {games_used} distinct games -- correlated risk)"
    print(f"\n{label}")
    print(f"  Joint probability (assumes independence): {jp:.1%}{diversity_note}")
    print("  " + "-" * 76)
    for p in squad:
        print(f"  {p['batter']:<22} {p['team']:<18} proj={p['projection']:.2f}  "
              f"P({metric_label}>{line})={p['prob']:.1%}   [{p['game']}]")


def main(date_str, team_filter, csv_path, metric, line, squad_size, num_alternatives, max_per_game):
    rows = load_rows(date_str, team_filter, csv_path)
    if not rows:
        print("No model rows available.")
        return

    metric_label = METRIC_LABEL[metric]
    candidates = compute_probabilities(rows, metric, line)
    if len(candidates) < squad_size:
        print(f"Only {len(candidates)} candidates found -- need at least {squad_size} for a squad. "
              f"Try a wider --team scope or check the metric field is present in your data.")
        return

    print(f"Ranked {len(candidates)} batters by P({metric_label} > {line}) for {date_str}\n")
    print("Top 15 individually:")
    for c in candidates[:15]:
        print(f"  {c['batter']:<22} {c['team']:<18} proj={c['projection']:.2f}  P={c['prob']:.1%}   [{c['game']}]")

    # Best possible squad: pure top-N, no diversification constraint --
    # mathematically maximizes the naive joint probability.
    best = build_squad(candidates, squad_size)
    print_squad(best, f"BEST SQUAD (top {squad_size} by probability, no game cap)", metric_label, line, squad_size)

    # Diversified squad: caps players per game to reduce correlated risk
    diversified = None
    if max_per_game:
        diversified = build_squad(candidates, squad_size, max_per_game=max_per_game)
        print_squad(diversified, f"DIVERSIFIED SQUAD (max {max_per_game}/game)", metric_label, line, squad_size)

    # Alternative squads: non-overlapping with what's shown above, so these are
    # genuinely different options rather than near-duplicates of the best squad.
    used = set(p["batter"] for p in best)
    if diversified:
        used.update(p["batter"] for p in diversified)

    print(f"\n{'=' * 80}\nALTERNATIVE SQUADS (no player overlap with squads above)\n{'=' * 80}")
    for i in range(num_alternatives):
        alt = build_squad(candidates, squad_size, max_per_game=max_per_game, exclude_batters=used)
        if len(alt) < squad_size:
            print(f"\nAlternative {i + 1}: not enough remaining candidates for a full squad.")
            break
        print_squad(alt, f"Alternative {i + 1}", metric_label, line, squad_size)
        used.update(p["batter"] for p in alt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build squads of players who must all clear a threshold together")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--team", default=None, help="Scope to specific teams, comma-separated")
    parser.add_argument("--csv", default=None, help="Reuse an existing mlb_report_*.csv instead of recomputing")
    parser.add_argument("--metric", choices=["combined", "total_bases"], required=True)
    parser.add_argument("--line", type=float, default=1.5, help="Typically 1.5 or 2.5")
    parser.add_argument("--squad-size", type=int, default=4)
    parser.add_argument("--num-alternatives", type=int, default=3)
    parser.add_argument("--max-per-game", type=int, default=2,
                         help="Cap players from the same game per squad, for diversification. 0 = no cap.")
    args = parser.parse_args()

    main(args.date, args.team, args.csv, args.metric, args.line, args.squad_size,
         args.num_alternatives, args.max_per_game if args.max_per_game > 0 else None)
