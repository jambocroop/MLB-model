"""
Odds Value Finder: match model projections against real sportsbook lines.
============================================================================

Everything so far measures ACCURACY (hit rate, correlation). This measures
PROFITABILITY POTENTIAL: given the actual odds a book is offering right now,
does the model's projection imply a real edge, or is the book already
pricing it efficiently?

Data source: The Odds API (the-odds-api.com) -- free tier, 500 credits/month.
    - MLB player props confirmed available: batter_hits, batter_total_bases,
      batter_rbis, batter_runs_scored, and batter_hits_runs_rbis (the exact
      combined prop this model projects as expected_combined).
    - Cost = [markets requested] x [regions] per game. Requesting all 5
      markets for 1 region ("us") = 5 credits/game. A full ~15-game slate
      is ~75 credits -- roughly 6-7 full-slate days/month on the free tier.
      Scope to specific games with --team to stretch the free quota further.
    - IMPORTANT: historical player prop odds require a PAID plan on this
      API. This script only works with LIVE/current odds -- it cannot
      retroactively backtest profitability against real historical lines.
      The honest path to validating profitability is prospective: run this
      daily going forward and track results, rather than backtesting.

METHODOLOGY:
    The model outputs point projections (e.g. expected_total_bases=1.8), not
    probabilities. To compare against a betting line ("Over 1.5"), a
    projection has to become P(actual > 1.5). This script converts using:
      - Hits: Binomial(n=at-bats-per-game, p=hit-probability-per-AB) -- we
        already have both components, and hits are naturally bounded by AB.
      - Total Bases / Runs / RBI / Combined: Poisson(lambda=projection) --
        a standard, defensible approximation for count-like sports stats
        when a proper fitted distribution isn't available. Not exact (total
        bases isn't purely Poisson), but reasonable for a first pass.

    Market probability is de-vigged from the two-sided price (Over price +
    Under price both reflect the book's margin; normalizing them to sum to
    1 gives a fairer "true" market probability estimate) where both sides
    are available, otherwise raw implied probability from the offered side.

    Edge = model_probability - market_probability (de-vigged when possible)
    EV   = model_probability x (decimal_odds - 1) - (1 - model_probability)
           expressed as expected return per $1 staked at the ACTUAL offered
           price (which includes the vig -- EV vs the raw price is what you
           actually experience placing the bet, edge above is diagnostic).

USAGE:
    export ODDS_API_KEY=your_key_here
    python odds_value_finder.py --date 2026-08-26
    python odds_value_finder.py --date 2026-08-26 --team "Dodgers,Yankees"  # cheaper on quota
    python odds_value_finder.py --date 2026-08-26 --markets batter_total_bases,batter_hits_runs_rbis
    python odds_value_finder.py --csv mlb_report_2026-08-26.csv  # reuse an existing model run instead of recomputing

OUTPUT:
    Console leaderboard of the best-edge bets found, sorted by EV, and a
    full CSV (odds_value_2026-08-26.csv) with every matched player/market.
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime

import requests

from mlb_daily_analysis import analyze_date

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_MARKETS = ["batter_hits", "batter_total_bases", "batter_rbis",
                    "batter_runs_scored", "batter_hits_runs_rbis"]

# Maps an Odds API market key to the model's corresponding projection field
# and which probability model to use for converting that projection into
# P(actual > line).
MARKET_TO_MODEL_FIELD = {
    "batter_hits": ("expected_hits", "binomial_hits"),
    "batter_total_bases": ("expected_total_bases", "poisson"),
    "batter_rbis": ("expected_rbi", "poisson"),
    "batter_runs_scored": ("expected_runs", "poisson"),
    "batter_hits_runs_rbis": ("expected_combined", "poisson"),
}

# Which markets have an empirical calibration curve available (see
# calibrate_probability() below). Only Combined and Total Bases have been
# backtested for this -- others pass through uncalibrated.
MARKET_TO_CALIBRATION_METRIC = {
    "batter_total_bases": "total_bases",
    "batter_hits_runs_rbis": "combined",
}


# ---------------------------------------------------------------------------
# Probability math
# ---------------------------------------------------------------------------

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_prob_over(line, lam):
    """P(X > line) for X ~ Poisson(lam). `line` is typically a .5 value (e.g. 1.5)."""
    threshold = math.floor(line) + 1  # "over 1.5" means X >= 2
    cumulative = sum(poisson_pmf(k, lam) for k in range(threshold))
    return max(0.0, min(1.0, 1 - cumulative))


def binomial_prob_over(line, n, p):
    """P(X > line) for X ~ Binomial(n, p)."""
    n = max(0, round(n))
    p = max(0.0, min(1.0, p))
    threshold = math.floor(line) + 1
    if threshold > n:
        return 0.0
    cumulative = sum(math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(threshold))
    return max(0.0, min(1.0, 1 - cumulative))


def model_prob_over(field_value, model_type, line, ab_per_game=None, hit_component=None):
    if model_type == "poisson":
        return poisson_prob_over(line, field_value)
    elif model_type == "binomial_hits":
        # Recover (n, p) from what we already have: field_value = expected_hits
        # = ab_per_game * hit_component. If ab_per_game wasn't passed, fall back
        # to a Poisson approximation on expected_hits instead.
        if ab_per_game and ab_per_game > 0:
            p = field_value / ab_per_game
            return binomial_prob_over(line, ab_per_game, p)
        return poisson_prob_over(line, field_value)
    raise ValueError(f"Unknown model_type {model_type}")


# ---------------------------------------------------------------------------
# Empirical probability calibration
# ---------------------------------------------------------------------------
#
# The raw Poisson-derived probability above is well-calibrated in the low-
# middle range but systematically OVERCONFIDENT at high probabilities --
# found by backtesting (2026-07-16 to 2026-08-25, n=9,792 candidates per
# metric): binning every candidate by predicted P(clear) and comparing
# against the ACTUAL clear rate in each bin showed the gap growing from
# near-zero around the 30th percentile to +14.6% (Combined) / +19.1%
# (Total Bases) overconfident at the top decile. This matters a lot in
# practice: both squad_builder.py (picks the top-N by probability each day)
# and odds_value_finder.py's value-bet detection (flags picks where model
# probability notably exceeds market) specifically select from this
# high-probability tail -- so both were showing inflated confidence exactly
# where it's most consequential. A direct simulation of squad_builder's
# actual top-4-per-day squads found the effect concretely: 92.8% average
# predicted joint-contributing probability vs 72.6% actual clear rate for
# those specific selected picks.
#
# Each entry is (avg_predicted_prob, actual_clear_rate) from one bin of the
# empirical calibration curve, sorted ascending. Only covers "combined" and
# "total_bases" (the two metrics squad_builder/value-bet detection actually
# use) -- Hits/Runs/RBI individual markets aren't calibrated here since we
# don't have that data; calibrate_probability() passes those through
# unchanged. Re-derive this curve periodically from a fresh backtest.
CALIBRATION_CURVE = {
    "combined": [
        (0.112, 0.223), (0.239, 0.254), (0.320, 0.317), (0.386, 0.335), (0.446, 0.369),
        (0.504, 0.427), (0.559, 0.459), (0.619, 0.533), (0.694, 0.558), (0.815, 0.670),
    ],
    "total_bases": [
        (0.061, 0.155), (0.152, 0.155), (0.213, 0.190), (0.265, 0.221), (0.319, 0.254),
        (0.375, 0.290), (0.438, 0.353), (0.513, 0.429), (0.612, 0.526), (0.792, 0.601),
    ],
}


def calibrate_probability(raw_prob, metric):
    """
    Corrects a raw model probability using the empirical calibration curve
    above (linear interpolation between observed bin points; clamped to the
    boundary bin's actual rate outside the observed range rather than
    extrapolating). If `metric` isn't in CALIBRATION_CURVE (e.g. individual
    Hits/Runs/RBI markets), returns raw_prob unchanged.
    """
    curve = CALIBRATION_CURVE.get(metric)
    if not curve:
        return raw_prob
    if raw_prob <= curve[0][0]:
        return curve[0][1]
    if raw_prob >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        x0, y0 = curve[i]
        x1, y1 = curve[i + 1]
        if x0 <= raw_prob <= x1:
            frac = (raw_prob - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + frac * (y1 - y0)
    return raw_prob  # unreachable given the boundary checks above


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------

def american_to_decimal(price):
    if price > 0:
        return 1 + price / 100
    return 1 + 100 / abs(price)


def implied_prob(price):
    decimal = american_to_decimal(price)
    return 1 / decimal


def devig_two_way(over_price, under_price):
    """Normalize two implied probabilities (which sum to >1 due to vig) to sum to 1."""
    p_over = implied_prob(over_price)
    p_under = implied_prob(under_price)
    total = p_over + p_under
    if total <= 0:
        return None, None
    return p_over / total, p_under / total


def expected_value(model_prob, price):
    """Expected return per $1 staked at the given American odds, if model_prob is correct."""
    decimal = american_to_decimal(price)
    return model_prob * (decimal - 1) - (1 - model_prob)


# ---------------------------------------------------------------------------
# Odds API calls
# ---------------------------------------------------------------------------

def odds_api_get(path, api_key, params=None):
    params = dict(params or {})
    params["apiKey"] = api_key
    resp = requests.get(f"{ODDS_API_BASE}{path}", params=params, timeout=20)
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"  [Odds API quota remaining: {remaining}]")
    if resp.status_code != 200:
        print(f"  [Odds API ERROR {resp.status_code}] {resp.text[:300]}")
        return None
    return resp.json()


def get_todays_mlb_events(api_key):
    return odds_api_get("/sports/baseball_mlb/events", api_key) or []


def get_event_player_odds(event_id, markets, api_key):
    return odds_api_get(
        f"/sports/baseball_mlb/events/{event_id}/odds",
        api_key,
        params={"regions": "us", "markets": ",".join(markets), "oddsFormat": "american"},
    )


def get_game_totals(api_key, regions="us"):
    """
    Fetch today's MLB game totals (the Over/Under total-runs line) for ALL games
    in ONE call. This uses the bulk "featured markets" endpoint (h2h/spreads/
    totals), NOT the per-event player-props endpoint -- costs just 1 credit
    (markets=1 x regions=1) for the entire day's slate, dramatically cheaper
    than player props. A higher total implies the market expects more scoring
    (offense-friendly park/weather/pitching matchups), useful as a signal for
    which GAMES to prioritize before picking which players within them.

    Returns a list of {home_team, away_team, total} dicts (consensus total,
    averaged across whichever books offered a totals line for that game).
    """
    data = odds_api_get(
        "/sports/baseball_mlb/odds", api_key,
        params={"regions": regions, "markets": "totals", "oddsFormat": "american"},
    )
    if not data:
        return []
    results = []
    for event in data:
        lines = []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("point") is not None:
                        lines.append(outcome["point"])
        if lines:
            results.append({
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "total": round(sum(lines) / len(lines), 2),
            })
    return results


# ---------------------------------------------------------------------------
# Name matching (odds API player names vs MLB Stats API batter names)
# ---------------------------------------------------------------------------

def normalize_name(name):
    name = name.lower().strip()
    for suffix in (" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace(".", "").replace("-", " ").strip()


def build_name_index(rows):
    """batter name (normalized) -> row, for matching against Odds API descriptions."""
    index = {}
    for r in rows:
        index[normalize_name(r["batter"])] = r
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_market_calibration(all_matches, warn_threshold=0.10, min_n=10):
    """
    Sanity check: for each market, compare the model's AVERAGE probability to
    the market's AVERAGE (de-vigged) probability across all matched players.

    This catches a specific, real failure mode (found the hard way): a bug or
    miscalibration in the model produces a systematic bias in one direction
    for an entire market. A genuine per-player edge should be scattered in
    both directions -- some players the model likes more than the market,
    some less -- and roughly cancel out on average. If instead the model's
    average sits far below (or above) the market's average across dozens of
    players in the same market, that's a strong signal something's wrong
    with the projection itself, not that real value was found everywhere.

    Returns the list of flagged market keys and prints a report; does not
    filter or alter any bets -- just flags markets worth distrusting.
    """
    by_market = {}
    for m in all_matches:
        by_market.setdefault(m["market"], []).append(m)

    print("\n" + "=" * 90)
    print("MODEL CALIBRATION CHECK  (model avg prob vs market avg prob, per market)")
    print("=" * 90)
    flagged_markets = []
    for market in sorted(by_market):
        rows = by_market[market]
        if len(rows) < min_n:
            print(f"{market:<24} n={len(rows):<4} [skipped -- fewer than {min_n} matches to judge]")
            continue
        avg_model = sum(r["model_prob"] for r in rows) / len(rows)
        avg_market = sum(r["market_prob"] for r in rows) / len(rows)
        gap = avg_model - avg_market
        flagged = abs(gap) >= warn_threshold
        status = "\u26a0 SUSPICIOUS" if flagged else "OK"
        if flagged:
            flagged_markets.append(market)
        print(f"{market:<24} n={len(rows):<4} model_avg={avg_model:.3f}  market_avg={avg_market:.3f}  "
              f"gap={gap:+.3f}  [{status}]")

    if flagged_markets:
        print(f"\n\u26a0 Systematic gap found in: {', '.join(flagged_markets)}")
        print("  A real per-player edge should roughly cancel out across many players in the")
        print("  same market -- a large SAME-DIRECTION gap usually means a bug or miscalibration")
        print("  in that projection, not real distributed value. Treat 'value bets' in flagged")
        print("  markets with real skepticism until the underlying projection is checked.")
    print()
    return flagged_markets


def find_value_bets(model_rows, api_key, markets, min_edge=0.05):
    name_index = build_name_index(model_rows)
    events = get_todays_mlb_events(api_key)
    if not events:
        print("No MLB events found from Odds API (wrong date, off-season, invalid API key, or API issue).")
        return [], []

    print(f"Found {len(events)} MLB event(s) from Odds API.\n")
    results = []

    for event in events:
        matchup = f"{event.get('away_team')} @ {event.get('home_team')}"
        print(f"Fetching odds: {matchup}")
        data = get_event_player_odds(event["id"], markets, api_key)
        if not data:
            continue

        for bookmaker in data.get("bookmakers", []):
            book_name = bookmaker.get("title", bookmaker.get("key"))
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                if market_key not in MARKET_TO_MODEL_FIELD:
                    continue
                field_name, model_type = MARKET_TO_MODEL_FIELD[market_key]

                # Group outcomes by (player, line) to pair Over/Under together
                by_player_line = {}
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description")
                    line = outcome.get("point")
                    side = outcome.get("name")  # "Over" / "Under"
                    price = outcome.get("price")
                    if not player or line is None or price is None:
                        continue
                    key = (player, line)
                    by_player_line.setdefault(key, {})[side] = price

                for (player, line), sides in by_player_line.items():
                    row = name_index.get(normalize_name(player))
                    if row is None:
                        continue  # not in today's model output (didn't play, no lineup, etc.)
                    if field_name not in row or row[field_name] is None:
                        continue

                    ab_per_game = None
                    if row.get("recent_ab") and row.get("recent_games"):
                        ab_per_game = row["recent_ab"] / row["recent_games"]

                    m_prob = model_prob_over(row[field_name], model_type, line, ab_per_game=ab_per_game)
                    m_prob = calibrate_probability(m_prob, MARKET_TO_CALIBRATION_METRIC.get(market_key))

                    over_price = sides.get("Over")
                    under_price = sides.get("Under")
                    if over_price is None:
                        continue

                    if under_price is not None:
                        market_prob_over, _ = devig_two_way(over_price, under_price)
                    else:
                        market_prob_over = implied_prob(over_price)

                    if market_prob_over is None:
                        continue

                    edge = m_prob - market_prob_over
                    ev = expected_value(m_prob, over_price)

                    results.append({
                        "batter": row["batter"], "team": row["team"], "opp_pitcher": row["opp_pitcher"],
                        "market": market_key, "line": line, "book": book_name,
                        "over_price": over_price, "under_price": under_price,
                        "model_prob": round(m_prob, 3), "market_prob": round(market_prob_over, 3),
                        "edge": round(edge, 3), "ev_per_dollar": round(ev, 3),
                        "note": row.get("note", ""),
                    })

    value_bets = [r for r in results if r["edge"] >= min_edge]
    value_bets.sort(key=lambda r: -r["ev_per_dollar"])
    return results, value_bets


def main(date_str, api_key, markets, team_filter, min_edge, csv_path):
    if csv_path:
        print(f"Loading existing model output from {csv_path}...")
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for key in ("expected_hits", "expected_total_bases", "expected_rbi",
                        "expected_runs", "expected_combined", "recent_ab", "recent_games"):
                if key in r and r[key] not in (None, ""):
                    r[key] = float(r[key])
    else:
        print(f"Running model for {date_str}...")
        rows, _ = analyze_date(date_str, team_filter=team_filter, use_statcast=False, workers=8)

    if not rows:
        print("No model rows to match against odds. Nothing to do.")
        return

    all_matches, value_bets = find_value_bets(rows, api_key, markets, min_edge)

    flagged_markets = check_market_calibration(all_matches) if all_matches else []

    print(f"\n{'=' * 90}")
    print(f"TOP VALUE BETS  (edge >= {min_edge:.0%}, sorted by EV per $1 staked)")
    print(f"{'=' * 90}")
    if not value_bets:
        print("No bets cleared the edge threshold. Either the market is efficient today, "
              "the name-matching missed some players, or try lowering --min-edge.")
    for b in value_bets[:20]:
        flag = "  \u26a0 FLAGGED MARKET" if b["market"] in flagged_markets else ""
        print(f"{b['batter']:<22} {b['market']:<22} line={b['line']:<5} @{b['book']:<12} "
              f"price={b['over_price']:>5}  model={b['model_prob']:.1%} vs market={b['market_prob']:.1%}  "
              f"edge={b['edge']:+.1%}  EV=${b['ev_per_dollar']:+.3f}{flag}")

    if all_matches:
        out_path = f"odds_value_{date_str}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_matches[0].keys()))
            writer.writeheader()
            writer.writerows(all_matches)
        print(f"\nFull matched odds ({len(all_matches)} rows) saved to {out_path}")
    else:
        print("\nNo odds could be matched to model output -- check that names align "
              "and that the requested markets are actually offered today.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match model projections against live sportsbook odds")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"),
                         help="The Odds API key. Defaults to $ODDS_API_KEY env var.")
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS),
                         help="Comma-separated Odds API market keys (default: all 5 batter props)")
    parser.add_argument("--team", default=None,
                         help="Scope model + odds lookups to specific teams, comma-separated -- "
                              "cheaper on Odds API quota than a full slate")
    parser.add_argument("--min-edge", type=float, default=0.05,
                         help="Minimum (model_prob - market_prob) to flag as a value bet (default: 0.05)")
    parser.add_argument("--csv", default=None,
                         help="Reuse an existing mlb_report_*.csv instead of recomputing the model")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("No Odds API key found. Set ODDS_API_KEY env var or pass --api-key. "
                 "Get a free key at https://the-odds-api.com/")

    main(args.date, args.api_key, args.markets.split(","), args.team, args.min_edge, args.csv)
