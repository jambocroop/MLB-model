"""
Pitcher Strikeouts Projection Model
====================================

STATUS: UNVALIDATED. This is a fresh build, following the exact same
tiered-blend architecture as the batter model in mlb_daily_analysis.py, but
for a completely different underlying stat with its own dynamics. It has
NOT been backtested. Every number this produces -- projections, the
Poisson-derived probability, everything -- should be treated as an
untested hypothesis until run through backtest_pitchers.py (build pending)
against real historical outcomes, the same discipline that caught real,
serious problems in the batter model (a systematic Runs bias, a BvP
overconfidence-on-thin-samples bug, and a probability-calibration problem
that would have gone undetected without exactly this kind of validation).
Don't skip that step just because the architecture is familiar.

WHY PITCHER STRIKEOUTS: multiple independent sources flagged this as one
of the more learnable MLB props for pick'em-style formats, and unlike the
batter props built so far, it has a genuinely new predictive signal
available: PITCHER WORKLOAD MANAGEMENT. A pitcher can have excellent
swing-and-miss stuff and still be capped on strikeout upside if he's being
kept to 5-6 innings regardless of performance (rest management, pitch-count
caps, September/postseason roster management, coming back from injury,
etc.) -- a real, observable pattern that has no batter-side equivalent in
this project. This model treats it as a first-class signal (INNINGS_TREND
flag below), not an afterthought.

METHODOLOGY (mirrors the batter model's tiered-blend approach):
    Priority chain per pitcher, each requiring a minimum sample:
      1. Pitcher-vs-opposing-team history (if a large enough sample exists)
      2. Recent form (last N starts)
      3. Fallback: recent form alone

    Both K/9-equivalent rate AND innings-per-start are blended, so the
    projection is (recent/matchup strikeout RATE) x (expected innings this
    start) -- not just a raw K/9 number, since innings pitched varies
    start-to-start and directly caps total strikeout opportunity.

    An innings-management flag surfaces separately (visible caution, not a
    silent multiplier -- same design principle as the batter model's PA
    reliability flag) when a pitcher's recent innings-per-start trend sits
    notably below his own season-long average -- a sign of workload
    management that could cap tonight's K ceiling regardless of stuff.

USAGE:
    python pitcher_strikeouts.py --date 2026-08-26
    python pitcher_strikeouts.py --date 2026-08-26 --team "Dodgers,Yankees"

OUTPUT:
    Console leaderboard + CSV, same pattern as mlb_daily_analysis.py.
"""

import argparse
import math
import time
from datetime import datetime, timedelta

from mlb_daily_analysis import api_get, get_todays_games, BASE

# ---------------------------------------------------------------------------
# Tunable constants -- all UNVALIDATED, carried over from the batter model's
# defaults as a reasonable starting point, not backtested for pitchers.
# ---------------------------------------------------------------------------
MIN_VS_TEAM_BF = 15          # minimum batters-faced vs this specific team to trust that tier
IP_SHRINKAGE_FULL_TRUST_STARTS = 5   # starts needed before fully trusting recent IP/start
RECENT_FORM_DAYS = 30         # starters pitch every ~5 days; 30 days ~ 5-6 starts


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def get_pitcher_recent_form(pitcher_id, end_date, days=RECENT_FORM_DAYS):
    """
    Pitcher's own recent starts, ending the day BEFORE end_date (same
    leakage-avoidance fix as the batter model's get_recent_form -- see that
    function's docstring for why this matters, especially for backtesting).
    """
    window_end = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    data = api_get(
        f"{BASE}/people/{pitcher_id}/stats",
        params={
            "stats": "byDateRange", "startDate": start_date, "endDate": window_end,
            "group": "pitching", "sportId": 1,
        },
    )
    return _extract_pitching_split(data)


def get_pitcher_season_form(pitcher_id, season):
    """Season-long pitching line -- used as the workload-management baseline
    (what's this pitcher's NORMAL innings-per-start, to compare recent trend against)."""
    data = api_get(
        f"{BASE}/people/{pitcher_id}/stats",
        params={"stats": "season", "season": season, "group": "pitching", "sportId": 1},
    )
    return _extract_pitching_split(data)


def get_pitcher_vs_team(pitcher_id, team_id, current_season, seasons=None):
    """
    Pitcher's history against this specific opposing team, built from his
    individual game log entries filtered to games started against that team.

    An earlier version used stats=vsTeam directly, which returned 0 BF for
    EVERY pitcher tested (~15 pitchers, including several long-tenured
    veterans certain to have faced these opponents many times) -- a uniform
    zero across that many pitchers meant the parameter/endpoint combination
    doesn't actually work for pitching stats, not that the samples were
    genuinely all empty. Rewritten to use gameLog instead, a more
    well-established endpoint (same reliability tier as byDateRange, which
    already works correctly elsewhere in this project).

    `seasons`: if given (list of ints), restricts to those seasons only --
    pass seasons strictly before the target date's year for leak-free
    backtesting (same pattern as the batter model's bvp_seasons). If not
    given (live use), defaults to the current season + 2 prior seasons, for
    a fuller sample than a single season alone would give.
    """
    seasons_to_check = seasons or [current_season, current_season - 1, current_season - 2]

    total_k, total_ip, total_bf, total_games = 0, 0.0, 0, 0
    for season in seasons_to_check:
        data = api_get(
            f"{BASE}/people/{pitcher_id}/stats",
            params={"stats": "gameLog", "season": season, "group": "pitching", "sportId": 1},
        )
        if not data:
            continue
        try:
            splits = data["stats"][0]["splits"]
        except (KeyError, IndexError, TypeError):
            continue
        for split in splits:
            opponent = split.get("opponent", {})
            if opponent.get("id") != team_id:
                continue
            stat = split.get("stat", {})
            total_k += int(stat.get("strikeOuts", 0) or 0)
            total_ip += _parse_innings(stat.get("inningsPitched", "0.0"))
            total_bf += int(stat.get("battersFaced", 0) or 0)
            total_games += 1

    if total_games == 0:
        return None
    return {
        "strikeouts": total_k, "innings_pitched": total_ip,
        "games_started": total_games, "batters_faced": total_bf, "era": None,
    }


def get_team_k_rate(team_id, season):
    """
    Opposing team's overall strikeout rate (season-long, hitting side) --
    context for how strikeout-prone this lineup is as a whole, independent
    of the specific pitcher facing them.
    """
    data = api_get(
        f"{BASE}/teams/{team_id}/stats",
        params={"stats": "season", "season": season, "group": "hitting", "sportId": 1},
    )
    if not data:
        return None
    try:
        stat = data["stats"][0]["splits"][0]["stat"]
        pa = float(stat.get("plateAppearances", 0) or 0)
        k = float(stat.get("strikeOuts", 0) or 0)
        if pa <= 0:
            return None
        return k / pa
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _extract_pitching_split(data):
    """Pull K/IP/games out of a pitching stats API response."""
    if not data:
        return None
    try:
        splits = data["stats"][0]["splits"]
        if not splits:
            return None
        stat = splits[0]["stat"]
        ip_str = stat.get("inningsPitched", "0.0")  # MLB reports "6.1" = 6 and 1/3 innings
        ip = _parse_innings(ip_str)
        return {
            "strikeouts": int(stat.get("strikeOuts", 0) or 0),
            "innings_pitched": ip,
            "games_started": int(stat.get("gamesStarted", stat.get("gamesPlayed", 0)) or 0),
            "batters_faced": int(stat.get("battersFaced", 0) or 0),
            "era": float(stat.get("era", 0) or 0) if stat.get("era") not in (None, "-", "") else None,
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _parse_innings(ip_str):
    """MLB's innings-pitched notation: '6.1' means 6 full innings + 1 out (=6.333...), '6.2' means 6 + 2 outs (6.667)."""
    try:
        ip_str = str(ip_str)
        if "." not in ip_str:
            return float(ip_str)
        whole, frac = ip_str.split(".")
        whole = float(whole)
        frac_outs = int(frac)  # 0, 1, or 2 outs into the next inning
        return whole + frac_outs / 3.0
    except (ValueError, TypeError):
        return 0.0


def _sum_pitching_splits(splits):
    if not splits:
        return None
    k = sum(s["strikeouts"] for s in splits)
    ip = sum(s["innings_pitched"] for s in splits)
    games = sum(s["games_started"] for s in splits)
    bf = sum(s["batters_faced"] for s in splits)
    return {
        "strikeouts": k, "innings_pitched": ip, "games_started": games,
        "batters_faced": bf, "era": None,  # ERA isn't meaningfully summable across seasons this way
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pitcher_strikeouts(vs_team, recent, season, min_vs_team_bf=MIN_VS_TEAM_BF):
    """
    Returns dict with expected_strikeouts, expected_innings, k_rate_component,
    innings_management_flag, and note explaining which tier was used.

    k_rate = strikeouts per inning pitched (a cleaner unit than per-batter-
    faced for this purpose, since it directly multiplies with expected
    innings to give expected strikeouts).
    """
    def k_per_ip(split):
        if not split or not split.get("innings_pitched"):
            return None
        return split["strikeouts"] / split["innings_pitched"]

    recent_k_rate = k_per_ip(recent) or 0.0
    vs_team_k_rate = k_per_ip(vs_team)
    vs_team_bf = vs_team["batters_faced"] if vs_team else 0

    if vs_team and vs_team_bf >= min_vs_team_bf and vs_team_k_rate is not None:
        k_rate = 0.6 * vs_team_k_rate + 0.4 * recent_k_rate
        note = f"vs-team sample trusted ({vs_team_bf} BF, {vs_team_k_rate:.2f} K/IP)"
    else:
        k_rate = recent_k_rate
        note = f"vs-team sample too small ({vs_team_bf} BF) -- recent form only"

    # --- Expected innings this start, with workload-management awareness ---
    recent_ip_per_start = None
    if recent and recent.get("games_started"):
        recent_ip_per_start = recent["innings_pitched"] / recent["games_started"]

    season_ip_per_start = None
    if season and season.get("games_started"):
        season_ip_per_start = season["innings_pitched"] / season["games_started"]

    # Shrink recent IP/start toward season baseline for the PROJECTION (same
    # shrinkage pattern as the batter model's AB fix -- a couple of short
    # outings shouldn't be extrapolated at full strength), but flag the raw
    # recent trend separately and visibly if it's notably below the pitcher's
    # own season norm -- that's the workload-management signal.
    innings_management_flag = False
    if recent_ip_per_start is not None and season_ip_per_start is not None and season_ip_per_start > 0:
        confidence = min(1.0, (recent.get("games_started") or 0) / IP_SHRINKAGE_FULL_TRUST_STARTS)
        expected_innings = confidence * recent_ip_per_start + (1 - confidence) * season_ip_per_start
        if recent_ip_per_start < season_ip_per_start * 0.85:  # recent trend notably below own norm
            innings_management_flag = True
    elif recent_ip_per_start is not None:
        expected_innings = recent_ip_per_start
    elif season_ip_per_start is not None:
        expected_innings = season_ip_per_start
    else:
        expected_innings = 5.0  # league-ish default for a starter, UNVALIDATED

    expected_strikeouts = round(expected_innings * k_rate, 2)

    return {
        "expected_strikeouts": expected_strikeouts,
        "expected_innings": round(expected_innings, 2),
        "k_per_ip": round(k_rate, 3),
        "innings_management_flag": innings_management_flag,
        "recent_ip_per_start": round(recent_ip_per_start, 2) if recent_ip_per_start is not None else None,
        "season_ip_per_start": round(season_ip_per_start, 2) if season_ip_per_start is not None else None,
        "note": note,
    }


def poisson_prob_over(line, lam):
    """UNCALIBRATED for this prop -- see module docstring. Reused Poisson
    approximation, same as the batter model's expected-count props."""
    if lam <= 0:
        return 1.0 if line < 0 else 0.0
    threshold = math.floor(line) + 1
    cumulative = 0.0
    for k in range(threshold):
        cumulative += math.exp(-lam) * (lam ** k) / math.factorial(k)
    return max(0.0, min(1.0, 1 - cumulative))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_date(date_str, team_filter=None, min_vs_team_bf=MIN_VS_TEAM_BF,
                  vs_team_seasons=None, final_games_only=False, verbose=True, delay=0.15):
    def log(msg):
        if verbose:
            print(msg)

    log(f"Fetching MLB schedule for {date_str}...")
    games = get_todays_games(date_str)
    if not games:
        log("No games found.")
        return []

    if final_games_only:
        games = [g for g in games if g.get("status") == "Final"]

    if team_filter:
        needles = [n.strip().lower() for n in team_filter.split(",") if n.strip()]
        games = [g for g in games if any(
            n in g["home_team_name"].lower() or n in g["away_team_name"].lower() for n in needles)]
        if not games:
            log(f"No games found matching --team '{team_filter}'.")
            return []

    season = int(date_str[:4])
    rows = []

    for g in games:
        matchups = [
            ("home", g["home_pitcher"], g["away_team_id"], g["away_team_name"], g["home_team_name"]),
            ("away", g["away_pitcher"], g["home_team_id"], g["home_team_name"], g["away_team_name"]),
        ]
        for side, pitcher, opp_team_id, opp_team_name, pitcher_team_name in matchups:
            pitcher_id = pitcher.get("id")
            pitcher_name = pitcher.get("fullName", "TBD")
            if not pitcher_id:
                continue

            recent = get_pitcher_recent_form(pitcher_id, date_str)
            season_form = get_pitcher_season_form(pitcher_id, season)
            vs_team = get_pitcher_vs_team(pitcher_id, opp_team_id, season, seasons=vs_team_seasons)
            opp_k_rate = get_team_k_rate(opp_team_id, season if not vs_team_seasons else vs_team_seasons[0])

            scores = score_pitcher_strikeouts(vs_team, recent, season_form, min_vs_team_bf)

            log(f"{pitcher_name} ({pitcher_team_name}) vs {opp_team_name}: "
                f"expected_K={scores['expected_strikeouts']}, {scores['note']}"
                f"{'  [WARN: innings management]' if scores['innings_management_flag'] else ''}")

            rows.append({
                "date": date_str, "gamePk": g["gamePk"],
                "game": f"{g['away_team_name']} @ {g['home_team_name']}",
                "pitcher_id": pitcher_id, "pitcher": pitcher_name, "team": pitcher_team_name,
                "opp_team": opp_team_name,
                "expected_strikeouts": scores["expected_strikeouts"],
                "expected_innings": scores["expected_innings"],
                "k_per_ip": scores["k_per_ip"],
                "innings_management_flag": scores["innings_management_flag"],
                "recent_ip_per_start": scores["recent_ip_per_start"],
                "season_ip_per_start": scores["season_ip_per_start"],
                "opp_team_k_rate": round(opp_k_rate, 3) if opp_k_rate else None,
                "note": scores["note"],
            })
            time.sleep(delay)

    return rows


def run(date_str, team_filter=None):
    rows = analyze_date(date_str, team_filter=team_filter)
    if not rows:
        return

    print("\n" + "=" * 78)
    print("TOP PROJECTED STRIKEOUTS  (UNVALIDATED -- see module docstring)")
    print("=" * 78)
    for r in sorted(rows, key=lambda x: -x["expected_strikeouts"])[:15]:
        flag = "  \u26a0 INNINGS MGMT" if r["innings_management_flag"] else ""
        print(f"  {r['pitcher']:<22} {r['team']:<18} vs {r['opp_team']:<18} "
              f"K={r['expected_strikeouts']:.2f}  IP={r['expected_innings']:.1f}{flag}")

    import csv
    out_path = f"pitcher_strikeouts_{date_str}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} pitcher projections to {out_path}")
    print("\nREMINDER: this model is UNVALIDATED. Do not trust expected_strikeouts")
    print("or any derived probability until backtested against real outcomes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitcher strikeouts projection (UNVALIDATED)")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--team", default=None, help="Scope to specific teams, comma-separated")
    args = parser.parse_args()
    run(args.date, args.team)
