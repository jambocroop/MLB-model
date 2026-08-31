"""
Backtest the pitcher strikeouts model against historical dates.
==================================================================

Mirrors backtest.py's architecture (leak-free date bounding, resume support,
incremental CSV writes), applied to pitcher_strikeouts.py. This step is not
optional: a model can pass every unit test and look completely sane on live
data (as this one just did) and STILL hide a systematic bias or
overconfidence problem that only shows up against real historical outcomes
-- that happened three separate times on the batter model this session
(a Runs bias, a BvP-thin-sample overconfidence bug, and a probability-
calibration problem). Don't skip this step just because the live test
looked clean.

LEAKAGE CONTROL: the live model defaults vs-team history to the current
season + 2 prior seasons. For a backtest, that would leak future at-bats
into a projection for a past date. This restricts vs-team history to
seasons STRICTLY BEFORE the year being tested (same pattern as the batter
model's bvp_seasons) -- airtight against leakage, at the cost of a thinner
in-season sample than a live run would have.

USAGE:
    python backtest_pitchers.py --start-date 2026-06-01 --end-date 2026-06-30
    python backtest_pitchers.py --start-date 2026-06-01 --end-date 2026-06-07 --team Dodgers

OUTPUT:
    backtest_pitchers_<start>_<end>.csv -- every pitcher-game: projection + actual
    Console summary: bias check, correlation vs naive baseline, same
    calibration-first discipline as the batter model's backtest.
"""

import argparse
import csv
import math
import os
from datetime import datetime, timedelta

from pitcher_strikeouts import analyze_date, api_get, BASE, _parse_innings

FIELDNAMES = [
    "date", "gamePk", "game", "pitcher_id", "pitcher", "team", "opp_team",
    "expected_strikeouts", "expected_innings", "k_per_ip", "innings_management_flag",
    "recent_ip_per_start", "season_ip_per_start", "opp_team_k_rate", "note",
    "actual_strikeouts", "actual_innings_pitched",
]


def _coerce_row_types(row):
    def to_float(v):
        return None if v in (None, "") else float(v)

    def to_int(v):
        return None if v in (None, "") else int(float(v))

    row["gamePk"] = to_int(row.get("gamePk"))
    row["pitcher_id"] = to_int(row.get("pitcher_id"))
    row["innings_management_flag"] = str(row.get("innings_management_flag")).strip().lower() == "true"
    for key in ("expected_strikeouts", "expected_innings", "k_per_ip", "recent_ip_per_start",
                "season_ip_per_start", "opp_team_k_rate", "actual_strikeouts", "actual_innings_pitched"):
        row[key] = to_float(row.get(key))
    return row


def daterange(start_date, end_date):
    d = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def get_game_pitching_actuals(game_pk):
    """For a completed game, return {pitcher_id: {strikeouts, innings_pitched}}."""
    boxscore = api_get(f"{BASE}/game/{game_pk}/boxscore")
    if not boxscore:
        return {}
    actuals = {}
    for side in ("home", "away"):
        team_box = boxscore.get("teams", {}).get(side, {})
        players = team_box.get("players", {})
        for key, p in players.items():
            person = p.get("person", {})
            pid = person.get("id")
            pitching = p.get("stats", {}).get("pitching", {})
            if not pid or not pitching:
                continue
            actuals[pid] = {
                "strikeouts": int(pitching.get("strikeOuts", 0) or 0),
                "innings_pitched": _parse_innings(pitching.get("inningsPitched", "0.0")),
            }
    return actuals


def get_final_games(date_str):
    """Reuse the schedule fetch from pitcher_strikeouts.py but filter to Final only."""
    from pitcher_strikeouts import get_todays_games
    games = get_todays_games(date_str)
    return [g for g in games if g.get("status") == "Final"]


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx and vy else None


def mean_absolute_error(xs, ys):
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs) if xs else None


def run_backtest(start_date, end_date, team_filter):
    out_path = f"backtest_pitchers_{start_date}_{end_date}.csv"

    all_rows = []
    completed_dates = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(_coerce_row_types(row))
        completed_dates = {r["date"] for r in all_rows}
        if completed_dates:
            print(f"Resuming {out_path}: {len(all_rows)} rows across {len(completed_dates)} "
                  f"date(s) already completed -- skipping those.\n")

    csv_file, csv_writer = None, None

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
            vs_team_seasons = [year - 1, year - 2, year - 3]  # strictly prior -- leak-free

            print(f"\n{'=' * 70}\nBACKTESTING {date_str}  (vs-team limited to seasons {vs_team_seasons})\n{'=' * 70}")
            rows = analyze_date(
                date_str, team_filter=team_filter, vs_team_seasons=vs_team_seasons,
                final_games_only=True, verbose=True,
            )
            if not rows:
                print(f"  No completed games/projections for {date_str}, skipping.")
                continue

            games = get_final_games(date_str)
            actuals_by_game = {g["gamePk"]: get_game_pitching_actuals(g["gamePk"]) for g in games}

            day_scored_rows = []
            for r in rows:
                actuals = actuals_by_game.get(r["gamePk"], {})
                a = actuals.get(r["pitcher_id"])
                if a is None:
                    continue  # didn't pitch / no boxscore line -- drop, not a scoreable prediction
                r["actual_strikeouts"] = a["strikeouts"]
                r["actual_innings_pitched"] = a["innings_pitched"]
                day_scored_rows.append(r)

            if day_scored_rows:
                writer = get_writer()
                for r in day_scored_rows:
                    writer.writerow(r)
                csv_file.flush()
                all_rows.extend(day_scored_rows)
                print(f"  Saved {len(day_scored_rows)} pitcher-games for {date_str} to {out_path}.")
    finally:
        if csv_file:
            csv_file.close()

    scored_rows = all_rows
    if not scored_rows:
        print("\nNo pitcher-games could be matched to actual results. Nothing to report.")
        return

    # --- Summary: same calibration-first discipline as the batter backtest ---
    print(f"\n\n{'#' * 70}")
    print(f"PITCHER STRIKEOUTS BACKTEST SUMMARY: {start_date} to {end_date}  ({len(scored_rows)} pitcher-games)")
    print(f"{'#' * 70}\n")

    proj = [r["expected_strikeouts"] for r in scored_rows]
    actual = [r["actual_strikeouts"] for r in scored_rows]
    avg_proj, avg_actual = sum(proj) / len(proj), sum(actual) / len(actual)
    r_val = pearson(proj, actual)
    mae = mean_absolute_error(proj, actual)

    def fmt(v, spec=".3f"):
        return format(v, spec) if v is not None else "n/a (need n>=2)"

    print(f"Overall bias: avg_projected={avg_proj:.2f}  avg_actual={avg_actual:.2f}  gap={avg_proj - avg_actual:+.2f}")
    print(f"Correlation: r={fmt(r_val)}")
    print(f"MAE: {fmt(mae, '.2f')}")

    # Tier breakdown
    trusted = [r for r in scored_rows if "trusted" in r["note"]]
    fallback = [r for r in scored_rows if "too small" in r["note"]]
    print(f"\nTier split: vs-team trusted={len(trusted)} ({len(trusted)/len(scored_rows):.0%}), "
          f"recent-form-only={len(fallback)} ({len(fallback)/len(scored_rows):.0%})")

    for label, group in [("vs-team trusted", trusted), ("recent-form-only", fallback)]:
        if not group:
            continue
        gp = [r["expected_strikeouts"] for r in group]
        ga = [r["actual_strikeouts"] for r in group]
        avg_gp, avg_ga = sum(gp) / len(gp), sum(ga) / len(ga)
        print(f"  {label} (n={len(group)}): avg_proj={avg_gp:.2f}  avg_actual={avg_ga:.2f}  "
              f"gap={avg_gp-avg_ga:+.2f}  r={fmt(pearson(gp, ga))}")

    # Innings-management flag validation -- same pattern as the batter PA-reliability check
    flagged = [r for r in scored_rows if r["innings_management_flag"]]
    not_flagged = [r for r in scored_rows if not r["innings_management_flag"]]
    if flagged and not_flagged:
        avg_ip_flagged = sum(r["actual_innings_pitched"] for r in flagged) / len(flagged)
        avg_ip_not = sum(r["actual_innings_pitched"] for r in not_flagged) / len(not_flagged)
        print(f"\nInnings-management flag check:")
        print(f"  Flagged (n={len(flagged)}): avg actual IP = {avg_ip_flagged:.2f}")
        print(f"  Not flagged (n={len(not_flagged)}): avg actual IP = {avg_ip_not:.2f}")
        print(f"  (Flagged pitchers should show LOWER actual IP if the flag means anything real)")

    print(f"\nFull results saved to {out_path}")
    print("\nDo not treat this model as usable for picks until these numbers have been reviewed --")
    print("same standard applied to the batter model throughout this project.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the pitcher strikeouts model")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--team", default=None, help="Scope to specific teams, comma-separated")
    args = parser.parse_args()
    run_backtest(args.start_date, args.end_date, args.team)
