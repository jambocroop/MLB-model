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
import math
import os
from datetime import datetime

from mlb_daily_analysis import analyze_date as analyze_date_batters
from pitcher_strikeouts import analyze_date as analyze_date_pitchers
from odds_value_finder import poisson_prob_over, get_game_totals, calibrate_probability

METRIC_FIELD = {
    "combined": "expected_combined",
    "total_bases": "expected_total_bases",
    "strikeouts": "expected_strikeouts",
}
METRIC_LABEL = {
    "combined": "Hits+Runs+RBIs",
    "total_bases": "Total Bases",
    "strikeouts": "Pitcher Strikeouts",
}
# Which "kind" of row each metric draws from (batter model vs pitcher model --
# two different data sources), and the name-field each kind's raw rows use.
# Internally, every PICK in this tool's candidate pool is stored under a
# generic "batter" key regardless of whether it's actually a batter or a
# pitcher -- this keeps all the existing squad-building/exclusion/printing
# logic working unchanged for both kinds of picks without a larger refactor.
METRIC_ROW_KIND = {"combined": "batter", "total_bases": "batter", "strikeouts": "pitcher"}
ROW_NAME_FIELD = {"batter": "batter", "pitcher": "pitcher"}


def load_rows(date_str, team_filter, csv_path, metrics):
    """
    Returns {"batter": [...], "pitcher": [...]}, only fetching the row kinds
    actually needed for the requested metrics -- a combined/total_bases-only
    run doesn't pay for a pitcher-model fetch it won't use, and vice versa.
    """
    needed_kinds = set(METRIC_ROW_KIND[m] for m in metrics)
    result = {"batter": [], "pitcher": []}

    if csv_path:
        # CSV reuse supports one row kind per file -- detect which by schema.
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return result
        kind = "pitcher" if "expected_strikeouts" in rows[0] else "batter"
        numeric_fields = (["expected_combined", "expected_total_bases"] if kind == "batter"
                          else ["expected_strikeouts", "expected_innings", "k_per_ip"])
        for r in rows:
            for key in numeric_fields:
                if r.get(key) not in (None, ""):
                    r[key] = float(r[key])
        result[kind] = rows
        return result

    if "batter" in needed_kinds:
        rows, _ = analyze_date_batters(date_str, team_filter=team_filter, use_statcast=False, workers=8)
        result["batter"] = rows
    if "pitcher" in needed_kinds:
        result["pitcher"] = analyze_date_pitchers(date_str, team_filter=team_filter, verbose=False)
    return result


def compute_probabilities(rows_by_kind, metrics, lines):
    """
    Builds the PICK POOL, not a player list: each row can contribute up to one
    entry PER METRIC requested (Combined, Total Bases, and/or Strikeouts),
    each with its own line and probability. The same real player legitimately
    produces multiple entries (one per applicable metric) -- that's intended,
    since a squad "pick" here really means (player, stat, line), and the same
    player can fill multiple slots via different stats. Strikeouts draws from
    a different underlying model (pitcher_strikeouts.py) than Combined/Total
    Bases (mlb_daily_analysis.py) -- rows_by_kind keeps them separate until
    they're merged into one combined pick pool here.

    rows_by_kind: {"batter": [...], "pitcher": [...]}, from load_rows()
    metrics: list of metric keys to include, e.g. ["combined", "total_bases", "strikeouts"]
    lines: dict {metric: line}
    """
    scored = []
    for metric in metrics:
        kind = METRIC_ROW_KIND[metric]
        name_field = ROW_NAME_FIELD[kind]
        field = METRIC_FIELD[metric]
        line = lines[metric]
        for r in rows_by_kind[kind]:
            proj = r.get(field)
            if proj is None or proj == "":
                continue
            raw_prob = poisson_prob_over(line, float(proj))
            # Empirical calibration correction (see odds_value_finder.py for the
            # full writeup) -- the raw Poisson probability is overconfident at
            # the high end for every metric checked so far, exactly where
            # squad-building selects from. NOTE: each curve was measured at one
            # specific line (1.5 for Combined/Total Bases, 5.5 for Strikeouts);
            # applied here regardless of the actual --line used as a reasonable
            # approximation, not separately validated per line.
            prob = calibrate_probability(raw_prob, metric)
            scored.append({
                "batter": r[name_field], "team": r["team"], "game": r.get("game", ""),
                "opp_pitcher": r.get("opp_pitcher", r.get("opp_team", "")),
                "metric": metric, "line": line,
                "projection": float(proj), "prob": prob, "note": r.get("note", ""),
            })
    scored.sort(key=lambda x: -x["prob"])
    return scored


def joint_probability(squad):
    """Naive independence assumption -- product of individual probabilities.
    Kept around for comparison against the correlation-adjusted version below."""
    p = 1.0
    for player in squad:
        p *= player["prob"]
    return p


# Empirically measured pairwise correlations (Pearson, on the binary "cleared
# the line" outcome), from backtest data 2026-07-16 to 2026-08-25:
#   - Same-game, different-player pairs: n=83,232 pairs per metric
#   - Same-player, both-stats pairs (his Combined pick vs his Total Bases pick,
#     same game): n=9,792
#   - Different-game pairs (sanity check): correlation -0.005, essentially 0 --
#     confirms independence is a reasonable assumption ACROSS games.
# The same-player number is dramatically higher than same-game-different-player,
# which makes sense: Total Bases literally feeds into Combined for one player's
# own at-bats, so their outcomes share the same underlying events. Same-game
# different-player correlation is real but much smaller than intuition might
# suggest -- individual at-bat variance dominates over the shared team-offense
# signal. Re-measure periodically from a fresh backtest; these aren't permanent.
SAME_GAME_CORRELATION = {"combined": 0.040, "total_bases": 0.013}
SAME_PLAYER_CORRELATION = 0.709


def get_pairwise_rho(pick_a, pick_b):
    """Empirical correlation between two picks' binary clear/no-clear outcomes."""
    if pick_a["batter"] == pick_b["batter"]:
        return SAME_PLAYER_CORRELATION  # can only differ by metric -- pool has 1 entry/metric/player
    if pick_a["game"] == pick_b["game"]:
        # Different metrics in the same game: use the higher (more conservative) of
        # the two same-game correlations rather than picking one arbitrarily.
        return max(SAME_GAME_CORRELATION.get(pick_a["metric"], 0.0),
                   SAME_GAME_CORRELATION.get(pick_b["metric"], 0.0))
    return 0.0  # different games: measured near 0, treat as independent


def _binary_joint_prob(p_x, p_y, rho):
    """P(X=1, Y=1) for correlated Bernoulli X, Y with given marginals and correlation,
    clamped to the Frechet bounds so an edge-case rho/probability combination can't
    produce an invalid (negative or >1) probability."""
    cov = rho * math.sqrt(max(0.0, p_x * (1 - p_x)) * max(0.0, p_y * (1 - p_y)))
    joint = p_x * p_y + cov
    lo, hi = max(0.0, p_x + p_y - 1.0), min(p_x, p_y)
    return max(lo, min(hi, joint))


def correlated_joint_probability(squad):
    """
    Correlation-adjusted joint probability, using the empirical rho values above
    instead of assuming independence. Approximation for squads with 3+ mutually
    correlated picks (this uses each new pick's SINGLE strongest correlation with
    an already-included pick, not a full multivariate model) -- but captures the
    dominant pairwise effects that matter at squad sizes around 4, and is exact
    for the common 2-correlated-picks case.
    """
    if not squad:
        return 1.0
    included = []
    joint = squad[0]["prob"]
    included.append(squad[0])
    for pick in squad[1:]:
        best_rho, best_prev = 0.0, None
        for prev in included:
            rho = get_pairwise_rho(prev, pick)
            if abs(rho) > abs(best_rho):
                best_rho, best_prev = rho, prev
        if best_prev is not None and best_rho != 0.0:
            p_x = best_prev["prob"]
            pair_joint = _binary_joint_prob(p_x, pick["prob"], best_rho)
            conditional_p = pair_joint / p_x if p_x > 0 else pick["prob"]
            joint *= max(0.0, min(1.0, conditional_p))
        else:
            joint *= pick["prob"]
        included.append(pick)
    return max(0.0, min(1.0, joint))


def match_game_total(game_str, totals):
    """
    Match a model row's 'game' string (format "Away @ Home", from analyze_date)
    against the Odds API's totals list. Team names can differ in format between
    sources -- "LA Dodgers" vs "Los Angeles Dodgers" -- so this tries substring
    containment first, then falls back to last-word nickname matching (handles
    abbreviated city names). Nickname matching alone risks collisions for teams
    sharing a last word (Red Sox / White Sox both end in "Sox"), so when more
    than one candidate matches by nickname, it requires a second shared word
    (the city/color/etc.) to disambiguate before accepting a match.
    """
    if " @ " not in game_str:
        return None
    away, home = [s.strip().lower() for s in game_str.split(" @ ", 1)]
    away_words, home_words = set(away.split()), set(home.split())

    # Pass 1: substring containment -- handles "Dodgers" being a suffix of
    # "Los Angeles Dodgers", or identical full names in different order/casing.
    for g in totals:
        odds_home, odds_away = g.get("home_team"), g.get("away_team")
        if not odds_home or not odds_away:
            continue
        oh, oa = odds_home.lower(), odds_away.lower()
        if (home in oh or oh in home) and (away in oa or oa in away):
            return g["total"]

    # Pass 2: last-word nickname match (catches "LA Dodgers" vs "Los Angeles
    # Dodgers", where neither string contains the other).
    def last_word(name):
        parts = name.strip().split()
        return parts[-1].lower() if parts else ""

    nickname_matches = []
    for g in totals:
        odds_home, odds_away = g.get("home_team"), g.get("away_team")
        if not odds_home or not odds_away:
            continue
        if last_word(odds_home) == last_word(home) and last_word(odds_away) == last_word(away):
            nickname_matches.append(g)

    if len(nickname_matches) == 1:
        return nickname_matches[0]["total"]
    elif len(nickname_matches) > 1:
        # Ambiguous by nickname alone (e.g. Red Sox vs White Sox both end in
        # "Sox") -- require a second shared word (the city/color/etc.) too.
        for g in nickname_matches:
            oh_words = set(g["home_team"].lower().split())
            oa_words = set(g["away_team"].lower().split())
            if (oh_words & home_words) and (oa_words & away_words):
                return g["total"]
    return None


def attach_game_totals(candidates, totals):
    matched, unmatched = 0, 0
    for c in candidates:
        total = match_game_total(c.get("game", ""), totals)
        c["game_total"] = total
        if total is not None:
            matched += 1
        else:
            unmatched += 1
    return matched, unmatched


def build_squad_by_game_priority(candidates, squad_size, max_per_game):
    """
    Prioritizes GAMES first (highest total-runs line = most expected offense),
    then picks the best player(s) within each game, up to max_per_game --
    rather than ranking players first and diversifying as an afterthought.
    Games with no matched total sort last (still usable, just deprioritized,
    so this degrades gracefully rather than excluding unmatched games).
    """
    by_game = {}
    for c in candidates:
        by_game.setdefault(c.get("game", ""), []).append(c)

    game_order = sorted(
        by_game.keys(),
        key=lambda g: (by_game[g][0].get("game_total") is None, -(by_game[g][0].get("game_total") or 0)),
    )

    squad = []
    for game in game_order:
        game_players = sorted(by_game[game], key=lambda c: -c["prob"])
        squad.extend(game_players[:max_per_game])
        if len(squad) >= squad_size:
            break
    squad.sort(key=lambda c: -c["prob"])
    return squad[:squad_size]


def build_squad(candidates, squad_size, max_per_game=None, exclude_picks=None):
    """Greedily take the highest-probability remaining candidates, respecting
    the per-game cap and any exclusions (used for building non-overlapping
    alternative squads). Exclusion is keyed by (batter, metric) -- a player
    can still be picked via the OTHER metric even if already used in a
    previous squad via one metric; that's intended, not an oversight."""
    exclude_picks = exclude_picks or set()
    game_counts = {}
    squad = []
    for c in candidates:
        if (c["batter"], c["metric"]) in exclude_picks:
            continue
        game = c.get("game", "")
        if max_per_game and game_counts.get(game, 0) >= max_per_game:
            continue
        squad.append(c)
        game_counts[game] = game_counts.get(game, 0) + 1
        if len(squad) == squad_size:
            break
    return squad


def print_squad(squad, label, squad_size):
    if len(squad) < squad_size:
        print(f"\n{label}: only {len(squad)}/{squad_size} candidates available -- not enough eligible players.")
        return
    jp_naive = joint_probability(squad)
    jp_adjusted = correlated_joint_probability(squad)
    games_used = len(set(p["game"] for p in squad))
    players_used = len(set(p["batter"] for p in squad))
    notes = []
    if players_used < len(squad):
        notes.append(f"only {players_used} distinct players -- same player picked for both stats at least once "
                      f"(measured correlation ~{SAME_PLAYER_CORRELATION:.2f}, same at-bats drive both outcomes)")
    elif games_used < len(squad):
        notes.append(f"only {games_used} distinct games (measured same-game correlation is real but modest, "
                      f"~0.01-0.04 -- see correlation-adjusted probability below)")
    diversity_note = f"  (\u26a0 {'; '.join(notes)})" if notes else ""
    print(f"\n{label}")
    print(f"  Joint probability -- naive independence:    {jp_naive:.1%}")
    print(f"  Joint probability -- correlation-adjusted:  {jp_adjusted:.1%}{diversity_note}")
    print("  " + "-" * 76)
    for p in squad:
        total_str = f"  O/U={p['game_total']}" if p.get("game_total") is not None else ""
        metric_label = METRIC_LABEL[p["metric"]]
        print(f"  {p['batter']:<22} {p['team']:<15} {metric_label:<16} proj={p['projection']:.2f}  "
              f"P(>{p['line']})={p['prob']:.1%}   [{p['game']}]{total_str}")


def main(date_str, team_filter, csv_path, metrics, lines, squad_size, num_alternatives, max_per_game, api_key):
    rows = load_rows(date_str, team_filter, csv_path, metrics)
    if not rows["batter"] and not rows["pitcher"]:
        print("No model rows available.")
        return

    candidates = compute_probabilities(rows, metrics, lines)
    if len(candidates) < squad_size:
        print(f"Only {len(candidates)} candidates found -- need at least {squad_size} for a squad. "
              f"Try a wider --team scope or check the metric field is present in your data.")
        return

    # Attach game totals if an API key is available -- cheap (1 credit for the
    # whole slate), used to prioritize higher-scoring-environment games.
    by_total_squad = None
    if api_key:
        print("Fetching game totals (Over/Under total runs) from Odds API...")
        totals = get_game_totals(api_key)
        matched, unmatched = attach_game_totals(candidates, totals)
        print(f"Matched game totals for {matched} candidates ({unmatched} unmatched).\n")
        if totals:
            print("Today's game totals (highest scoring environment first):")
            for g in sorted(totals, key=lambda g: -g["total"]):
                print(f"  {g['away_team']} @ {g['home_team']}: O/U {g['total']}")
    else:
        for c in candidates:
            c["game_total"] = None
        print("No Odds API key provided -- skipping game-total prioritization "
              "(set ODDS_API_KEY or pass --api-key to enable).\n")

    lines_str = ", ".join(f"{METRIC_LABEL[m]}>{lines[m]}" for m in metrics)
    print(f"\nRanked {len(candidates)} picks ({lines_str}) for {date_str}\n")
    print("Top 15 individually (mixed across both stats):")
    for c in candidates[:15]:
        total_str = f"  O/U={c['game_total']}" if c.get("game_total") is not None else ""
        print(f"  {c['batter']:<22} {c['team']:<15} {METRIC_LABEL[c['metric']]:<16} proj={c['projection']:.2f}  "
              f"P={c['prob']:.1%}   [{c['game']}]{total_str}")

    # Best possible squad: pure top-N across the mixed pool, no diversification
    # constraint -- mathematically maximizes the naive joint probability.
    best = build_squad(candidates, squad_size)
    print_squad(best, f"BEST SQUAD (top {squad_size} picks by probability, no cap)", squad_size)

    # Diversified squad: caps picks per game to reduce correlated risk
    diversified = None
    if max_per_game:
        diversified = build_squad(candidates, squad_size, max_per_game=max_per_game)
        print_squad(diversified, f"DIVERSIFIED SQUAD (max {max_per_game}/game)", squad_size)

    # Game-total-prioritized squad: highest-scoring-environment games first,
    # best pick(s) within each.
    if api_key:
        by_total_squad = build_squad_by_game_priority(candidates, squad_size, max_per_game or 1)
        print_squad(by_total_squad, "GAME-TOTAL PRIORITIZED SQUAD (highest O/U games first)", squad_size)

    # Alternative squads: non-overlapping (by player+metric) with what's shown
    # above, so these are genuinely different options, not near-duplicates.
    used = set((p["batter"], p["metric"]) for p in best)
    if diversified:
        used.update((p["batter"], p["metric"]) for p in diversified)
    if by_total_squad:
        used.update((p["batter"], p["metric"]) for p in by_total_squad)

    print(f"\n{'=' * 80}\nALTERNATIVE SQUADS (no pick overlap with squads above)\n{'=' * 80}")
    for i in range(num_alternatives):
        alt = build_squad(candidates, squad_size, max_per_game=max_per_game, exclude_picks=used)
        if len(alt) < squad_size:
            print(f"\nAlternative {i + 1}: not enough remaining candidates for a full squad.")
            break
        print_squad(alt, f"Alternative {i + 1}", squad_size)
        used.update((p["batter"], p["metric"]) for p in alt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build squads of picks (player + stat) who must all clear a threshold together. "
                     "By default mixes BOTH Combined and Total Bases picks in one pool -- the same "
                     "player can fill two slots if both his picks rank highly, that's intended.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--team", default=None, help="Scope to specific teams, comma-separated")
    parser.add_argument("--csv", default=None, help="Reuse an existing mlb_report_*.csv or "
                         "pitcher_strikeouts_*.csv instead of recomputing (one row-kind per file)")
    parser.add_argument("--metric", choices=["combined", "total_bases", "strikeouts", "both", "all"],
                         default="both",
                         help="Which stat(s) to pull picks from. 'both' = combined+total_bases "
                              "(default, batter-only). 'all' = combined+total_bases+strikeouts "
                              "(mixes in pitcher strikeouts too). All picks land in one mixed pool.")
    parser.add_argument("--line", type=float, default=None,
                         help="Line to use for ALL requested metrics if they should share one (e.g. --line 1.5). "
                              "For different lines per metric, use --combined-line / --total-bases-line / "
                              "--strikeouts-line instead.")
    parser.add_argument("--combined-line", type=float, default=1.5, help="Line for Hits+Runs+RBIs (default 1.5)")
    parser.add_argument("--total-bases-line", type=float, default=1.5, help="Line for Total Bases (default 1.5)")
    parser.add_argument("--strikeouts-line", type=float, default=5.5,
                         help="Line for Pitcher Strikeouts (default 5.5, a common real prop line -- "
                              "also what the calibration curve was measured at)")
    parser.add_argument("--squad-size", type=int, default=4)
    parser.add_argument("--num-alternatives", type=int, default=3)
    parser.add_argument("--max-per-game", type=int, default=2,
                         help="Cap picks from the same game per squad, for diversification. 0 = no cap.")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"),
                         help="Odds API key, for game-total prioritization. Defaults to $ODDS_API_KEY. "
                              "Optional -- without it, everything still works, just without the "
                              "game-total-prioritized squad.")
    args = parser.parse_args()

    if args.metric == "both":
        metrics = ["combined", "total_bases"]
    elif args.metric == "all":
        metrics = ["combined", "total_bases", "strikeouts"]
    else:
        metrics = [args.metric]

    if args.line is not None:
        lines = {m: args.line for m in metrics}
    else:
        lines = {"combined": args.combined_line, "total_bases": args.total_bases_line,
                  "strikeouts": args.strikeouts_line}

    main(args.date, args.team, args.csv, metrics, lines, args.squad_size,
         args.num_alternatives, args.max_per_game if args.max_per_game > 0 else None, args.api_key)
