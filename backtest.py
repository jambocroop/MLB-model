"""
Backtest the MLB daily model against historical dates.
=======================================================

For each date in a range, this:
  1. Runs the same prediction engine as mlb_daily_analysis.py, but scoped to
     only games with Final status (real results exist).
  2. Pulls each batter's ACTUAL result from that game (H, R, RBI).
  3. Joins predictions to actuals and reports:
       - Top-decile hit rate: of the batters the model ranked in the top
         10% of hit_score each day, what fraction actually got a hit? Same
         for run_score/rbi_score. Compared against the baseline (all-batter)
         rate.
       - Correlation: Pearson correlation between each 0-100 score and the
         actual binary outcome, across every batter-game in the window.
       - Combined projection accuracy: correlation AND mean absolute error
         (MAE) between expected_combined (the projected Hits+Runs+RBIs
         count -- see mlb_daily_analysis.py) and the batter's actual
         combined count that game. This is the metric that matters most for
         validating the "H+R+RBI" prop-style projection specifically, since
         it's a continuous count, not a hit/miss score.

LEAKAGE CONTROL (important):
    Recent form and Statcast are naturally date-bounded already (they only
    look at data before the date being tested). BvP and hand-split are NOT
    date-bounded by default in mlb_daily_analysis.py (they use "as of right
    now" totals), which would leak future at-bats into a backtest.

    This script fixes that by restricting BvP/hand-split lookups to seasons
    STRICTLY BEFORE the year of the date being tested (`bvp_seasons`).
    That's airtight -- zero leakage -- but it means backtested BvP samples
    won't include any at-bats from the same season as the test date, even
    ones from earlier in that season. Expect thinner BvP samples (more
    fallback to Statcast-similar/hand-split/recent-form) than a live run
    would show. This is a real tradeoff, not a bug -- correctness over
    sample size for validation purposes.

USAGE:
    python backtest.py --start-date 2026-08-01 --end-date 2026-08-14
    python backtest.py --start-date 2026-08-01 --end-date 2026-08-14 --use-statcast  # opt in, slower
    python backtest.py --start-date 2026-08-01 --end-date 2026-08-14 --team "Dodgers,Yankees"

OUTPUT:
    backtest_<start>_<end>.csv       -- every batter-game: prediction + actual
    Console summary with hit-rate, correlation, and combined-projection MAE
"""

import argparse
import csv
import math
import os
from datetime import datetime, timedelta

from mlb_daily_analysis import analyze_date, get_game_batting_actuals

FIELDNAMES = [
    "date", "gamePk", "game", "batter_id", "batter", "team", "lineup_slot",
    "opp_pitcher", "pitcher_hand",
    "bvp_ab", "bvp_avg", "bvp_slg", "bvp_runs_rate", "bvp_rbi_rate",
    "statcast_similar_ab", "statcast_similar_avg", "statcast_similar_slg",
    "statcast_n_similar_pitchers",
    "hand_split_ab", "hand_split_avg", "hand_split_slg", "hand_split_runs_rate", "hand_split_rbi_rate",
    "recent_ab", "recent_games", "recent_avg", "recent_slg", "recent_runs_rate", "recent_rbi_rate",
    "hit_score", "run_score", "rbi_score", "combined_score",
    "expected_hits", "expected_runs", "expected_rbi", "expected_combined",
    "expected_total_bases", "pa_reliability", "low_pa_flag",
    "combined_confidence", "total_bases_confidence", "note",
    "actual_ab", "actual_h", "actual_r", "actual_rbi", "actual_combined", "actual_total_bases",
]


def _coerce_row_types(row):
    """CSV round-trips everything as strings -- convert the numeric fields
    back so a resumed run's old rows behave the same as freshly computed ones."""
    def to_float(v):
        return None if v in (None, "") else float(v)

    def to_int(v):
        return None if v in (None, "") else int(float(v))

    for key in ("gamePk", "batter_id", "lineup_slot", "statcast_similar_ab",
                "statcast_n_similar_pitchers", "hand_split_ab", "recent_ab", "recent_games",
                "actual_ab", "actual_h", "actual_r", "actual_rbi", "actual_combined", "actual_total_bases"):
        row[key] = to_int(row.get(key))
    row["bvp_ab"] = to_int(row.get("bvp_ab")) or 0
    row["low_pa_flag"] = str(row.get("low_pa_flag")).strip().lower() == "true"
    for key in ("bvp_avg", "bvp_slg", "bvp_runs_rate", "bvp_rbi_rate",
                "statcast_similar_avg", "statcast_similar_slg",
                "hand_split_avg", "hand_split_slg", "hand_split_runs_rate", "hand_split_rbi_rate",
                "recent_avg", "recent_slg", "recent_runs_rate", "recent_rbi_rate",
                "pa_reliability",
                "hit_score", "run_score", "rbi_score", "combined_score",
                "expected_hits", "expected_runs", "expected_rbi", "expected_combined", "expected_total_bases"):
        row[key] = to_float(row.get(key))
    return row


def daterange(start_date, end_date):
    d = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def pearson_corr(xs, ys):
    """Plain-Python Pearson correlation, no numpy/scipy dependency."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def mean_absolute_error(xs, ys):
    n = len(xs)
    if n == 0:
        return None
    return sum(abs(x - y) for x, y in zip(xs, ys)) / n


def top_decile_hit_rate(rows, score_key, actual_key):
    """
    For each date in `rows`, take the top 10% of batters by score_key
    (min 1 batter), check what fraction of them had actual_key > 0.
    Returns (top_decile_rate, baseline_rate, n_top, n_total).
    """
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)

    top_hits, top_total = 0, 0
    base_hits, base_total = 0, 0

    for date_str, day_rows in by_date.items():
        day_rows_sorted = sorted(day_rows, key=lambda r: -r[score_key])
        n_top = max(1, round(len(day_rows_sorted) * 0.10))
        top_group = day_rows_sorted[:n_top]

        for r in top_group:
            top_total += 1
            if r[actual_key] > 0:
                top_hits += 1
        for r in day_rows_sorted:
            base_total += 1
            if r[actual_key] > 0:
                base_hits += 1

    top_rate = top_hits / top_total if top_total else None
    base_rate = base_hits / base_total if base_total else None
    return top_rate, base_rate, top_total, base_total


def run_backtest(start_date, end_date, use_statcast, min_bvp_ab, recent_days,
                  statcast_lookback_days, similarity_threshold, team_filter, delay, workers=8):
    out_path = f"backtest_{start_date}_{end_date}.csv"

    all_rows = []
    completed_dates = set()

    # --- Resume support: if this exact backtest was already partially run,
    # pick up where it left off instead of redoing (and re-hitting the API
    # for) dates that are already saved. ---
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(_coerce_row_types(row))
        completed_dates = {r["date"] for r in all_rows}
        if completed_dates:
            print(f"Resuming {out_path}: {len(all_rows)} rows across {len(completed_dates)} "
                  f"date(s) already completed -- skipping those.\n")

    csv_file = None
    csv_writer = None

    def get_writer():
        nonlocal csv_file, csv_writer
        if csv_writer is None:
            is_new = not (os.path.exists(out_path) and os.path.getsize(out_path) > 0)
            csv_file = open(out_path, "a", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            if is_new:
                csv_writer.writeheader()
        return csv_writer

    try:
        for date_str in daterange(start_date, end_date):
            if date_str in completed_dates:
                print(f"Skipping {date_str} -- already completed (resume).")
                continue

            year = int(date_str[:4])
            bvp_seasons = [year - 1, year - 2]  # strictly prior seasons only -- leak-free

            print(f"\n{'=' * 70}\nBACKTESTING {date_str}  (BvP/hand-split limited to seasons {bvp_seasons})\n{'=' * 70}")
            rows, games = analyze_date(
                date_str,
                min_bvp_ab=min_bvp_ab,
                recent_days=recent_days,
                delay=delay,
                use_statcast=use_statcast,
                statcast_lookback_days=statcast_lookback_days,
                similarity_threshold=similarity_threshold,
                team_filter=team_filter,
                bvp_seasons=bvp_seasons,
                final_games_only=True,
                verbose=True,
                workers=workers,
            )
            if not rows:
                print(f"  No completed games/predictions for {date_str}, skipping.")
                continue

            # Pull actual results per game (one boxscore call per game, cached across batters)
            actuals_by_game = {}
            for g in games:
                if g.get("status") != "Final":
                    continue
                actuals_by_game[g["gamePk"]] = get_game_batting_actuals(g["gamePk"])

            day_scored_rows = []
            for r in rows:
                actuals = actuals_by_game.get(r["gamePk"], {})
                a = actuals.get(r["batter_id"])
                if a is None:
                    continue  # batter didn't end up playing / no boxscore line -- drop, not a prediction we can score
                r["actual_ab"] = a["ab"]
                r["actual_h"] = a["h"]
                r["actual_r"] = a["r"]
                r["actual_rbi"] = a["rbi"]
                r["actual_combined"] = a["h"] + a["r"] + a["rbi"]
                r["actual_total_bases"] = a["tb"]
                day_scored_rows.append(r)

            if day_scored_rows:
                writer = get_writer()
                for r in day_scored_rows:
                    writer.writerow(r)
                csv_file.flush()
                all_rows.extend(day_scored_rows)
                print(f"  Saved {len(day_scored_rows)} batter-games for {date_str} to {out_path}.")
    finally:
        if csv_file:
            csv_file.close()

    scored_rows = all_rows  # everything in all_rows already has actual results by construction

    if not scored_rows:
        print("\nNo batter-games could be matched to actual results. Nothing to report.")
        return

    # --- Metrics ---
    print(f"\n\n{'#' * 70}")
    print(f"BACKTEST SUMMARY: {start_date} to {end_date}  ({len(scored_rows)} batter-games)")
    print(f"{'#' * 70}\n")

    for label, score_key, actual_key in [
        ("HITS", "hit_score", "actual_h"),
        ("RUNS", "run_score", "actual_r"),
        ("RBIs", "rbi_score", "actual_rbi"),
    ]:
        top_rate, base_rate, n_top, n_base = top_decile_hit_rate(scored_rows, score_key, actual_key)
        xs = [r[score_key] for r in scored_rows]
        ys = [1 if r[actual_key] > 0 else 0 for r in scored_rows]
        corr = pearson_corr(xs, ys)

        print(f"--- {label} ---")
        if top_rate is not None and base_rate is not None:
            lift = (top_rate / base_rate - 1) * 100 if base_rate > 0 else None
            print(f"  Top-10% predicted ({n_top} picks): {top_rate:.1%} actually recorded a {label.lower()[:-1] if label != 'RUNS' else 'run'}")
            print(f"  Baseline (all {n_base} batters):     {base_rate:.1%}")
            if lift is not None:
                print(f"  Lift: {lift:+.1f}%")
        if corr is not None:
            print(f"  Correlation (score vs actual outcome): {corr:.3f}")
        print()

    def report_projection_block(label, proj_key, actual_key, event_label):
        print(f"--- {label} ---")
        top_rate, base_rate, n_top, n_base = top_decile_hit_rate(scored_rows, proj_key, actual_key)
        if top_rate is not None and base_rate is not None:
            lift = (top_rate / base_rate - 1) * 100 if base_rate > 0 else None
            print(f"  Top-10% projected ({n_top} picks): {top_rate:.1%} recorded at least 1 {event_label}")
            print(f"  Baseline (all {n_base} batters):    {base_rate:.1%}")
            if lift is not None:
                print(f"  Lift: {lift:+.1f}%")

        proj = [r[proj_key] for r in scored_rows]
        actual = [r[actual_key] for r in scored_rows]
        corr = pearson_corr(proj, actual)
        mae = mean_absolute_error(proj, actual)
        avg_proj = sum(proj) / len(proj)
        avg_actual = sum(actual) / len(actual)
        if corr is not None:
            print(f"  Correlation (projected vs actual count): {corr:.3f}")
        if mae is not None:
            print(f"  Mean absolute error: {mae:.2f}  (avg projected {avg_proj:.2f} vs avg actual {avg_actual:.2f})")
        print()

    # --- Combined projection (the real target: expected_combined ~ actual H+R+RBI count) ---
    report_projection_block("COMBINED (projected H+R+RBI count vs actual)",
                             "expected_combined", "actual_combined", "combined H/R/RBI event")

    # --- Total Bases projection ---
    report_projection_block("TOTAL BASES (projected vs actual)",
                             "expected_total_bases", "actual_total_bases", "total base")

    print(f"Full batter-by-batter results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the MLB daily model against historical dates")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--min-bvp-ab", type=int, default=8)
    parser.add_argument("--recent-days", type=int, default=15)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--use-statcast", action="store_true",
                         help="Opt in to Statcast pitcher-similarity (OFF by default -- a head-to-head "
                              "backtest, 2026-08-18 to 2026-08-24 n=1158, found it performs slightly worse "
                              "than the simpler hand-split fallback on both Hits and Total Bases, and it's "
                              "much slower. Kept as opt-in in case a stricter --similarity-threshold helps.)")
    parser.add_argument("--statcast-lookback-days", type=int, default=395)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--team", type=str, default=None,
                         help="Scope to specific teams each day, comma-separated for multiple, "
                              "e.g. --team \"Dodgers,Yankees\". Useful for a fast test run.")
    parser.add_argument("--workers", type=int, default=8,
                         help="Number of batters to analyze concurrently per day (default: 8). "
                              "This is the main lever for backtest speed -- a 2-week backtest that "
                              "took 30 min sequentially should drop dramatically with this. "
                              "Statcast calls are capped at min(workers, 4) internally regardless.")
    args = parser.parse_args()

    run_backtest(
        args.start_date, args.end_date,
        use_statcast=args.use_statcast,
        min_bvp_ab=args.min_bvp_ab,
        recent_days=args.recent_days,
        statcast_lookback_days=args.statcast_lookback_days,
        similarity_threshold=args.similarity_threshold,
        team_filter=args.team,
        delay=args.delay,
        workers=args.workers,
    )
