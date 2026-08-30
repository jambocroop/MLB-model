"""
MLB Daily High-Probability Outcome Finder
==========================================

Scans all MLB games scheduled for a given day and ranks batters by their
likelihood of recording a Hit, Run, or RBI, based on:

  1. Historical batter-vs-pitcher (BvP) performance  <- weighted heaviest
  2. Statcast pitcher-similarity performance -- when direct BvP sample is
     too small, find pitchers with a similar Statcast arsenal (pitch mix,
     velo, movement) that this batter HAS faced, and use his performance
     against that similar-pitcher group as a proxy
  3. Recent form (last N days of games)
  4. Batter-vs-pitcher-hand splits (last-resort fallback if neither BvP
     nor a Statcast-similar sample exists)
  5. Lineup slot (used only to weight Runs vs RBI likelihood -- leadoff/
     2-hole hitters score more runs, 3-4-5 hitters drive in more runs)

Data sources:
    - MLB Stats API (statsapi.mlb.com) -- free, no API key required.
    - Baseball Savant Statcast data via the `pybaseball` package.

USAGE:
    python mlb_daily_analysis.py                     # today's games
    python mlb_daily_analysis.py --date 2026-08-22
    python mlb_daily_analysis.py --min-bvp-ab 8       # min ABs to trust BvP
    python mlb_daily_analysis.py --use-statcast       # opt in to Statcast (slower, backtested as slightly worse)
    python mlb_daily_analysis.py --similarity-threshold 0.9

OUTPUT:
    Prints a ranked report to the console and saves a CSV to
    mlb_report_<date>.csv

NOTE: This was written and reviewed without live network access. The MLB
Stats API is undocumented/community-reverse-engineered, so if a call fails,
check the printed error -- it will usually show the exact URL and response,
which makes it easy to adjust a param name (see NOTES_ON_API_QUIRKS.md).
pybaseball scrapes Baseball Savant, so Statcast calls are much slower than
the MLB Stats API calls -- expect a full slate to take a while the first
time (pybaseball's on-disk cache speeds up repeat runs significantly).
"""

import argparse
import concurrent.futures
import csv
import math
import sys
import threading
import time
from datetime import datetime, timedelta

import requests

try:
    import pandas as pd
    import pybaseball as pyb
    pyb.cache.enable()  # cache Statcast pulls to disk between runs
    STATCAST_AVAILABLE = True
except ImportError:
    STATCAST_AVAILABLE = False

BASE = "https://statsapi.mlb.com/api/v1"
BASE_V11 = "https://statsapi.mlb.com/api/v1.1"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-daily-analysis/1.0"})


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------

def api_get(url, params=None, retries=2):
    """GET with basic retry + error surfacing."""
    for attempt in range(retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries:
                print(f"  [API ERROR] {url} params={params} -> {e}")
                return None
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Schedule / probable pitchers
# ---------------------------------------------------------------------------

def get_todays_games(date_str):
    """Return list of dicts: gamePk, home/away team id+name, probable pitchers."""
    data = api_get(
        f"{BASE}/schedule",
        params={
            "sportId": 1,
            "date": date_str,
            "hydrate": "team,probablePitcher",
        },
    )
    if not data:
        return []

    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            games.append({
                "gamePk": g["gamePk"],
                "status": g.get("status", {}).get("detailedState"),
                "home_team_id": home.get("team", {}).get("id"),
                "home_team_name": home.get("team", {}).get("name"),
                "away_team_id": away.get("team", {}).get("id"),
                "away_team_name": away.get("team", {}).get("name"),
                "home_pitcher": home.get("probablePitcher", {}),
                "away_pitcher": away.get("probablePitcher", {}),
            })
    return games


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------

def get_live_lineup(game_pk, team_side):
    """
    Try to pull the actual posted lineup from the live game feed.
    Only available once MLB posts lineups (~1-3 hrs before first pitch).
    Returns list of (player_id, full_name, batting_order_slot) or [].
    """
    data = api_get(f"{BASE_V11}/game/{game_pk}/feed/live")
    if not data:
        return []
    try:
        team_box = data["liveData"]["boxscore"]["teams"][team_side]
    except (KeyError, TypeError):
        return []

    order = team_box.get("battingOrder", [])
    players = team_box.get("players", {})
    lineup = []
    for slot_idx, pid in enumerate(order):
        p = players.get(f"ID{pid}", {})
        person = p.get("person", {})
        lineup.append((person.get("id"), person.get("fullName"), slot_idx + 1))
    return lineup


def get_fallback_lineup(team_id, before_date, num_games=5):
    """
    If no live lineup is posted yet, approximate the lineup using whichever
    9 position players started most often for this team in their last
    `num_games` games. Batting-order slot is approximated by average slot.
    """
    end = before_date
    start = (datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=20)).strftime("%Y-%m-%d")

    data = api_get(
        f"{BASE}/schedule",
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start,
            "endDate": end,
            "hydrate": "game",
        },
    )
    if not data:
        return []

    game_pks = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            if g.get("status", {}).get("statusCode") == "F":  # Final games only
                game_pks.append(g["gamePk"])
    game_pks = game_pks[-num_games:]

    appearances = {}  # player_id -> [name, [slots]]
    for pk in game_pks:
        boxscore = api_get(f"{BASE}/game/{pk}/boxscore")
        if not boxscore:
            continue
        for side in ("home", "away"):
            team_box = boxscore.get("teams", {}).get(side, {})
            if team_box.get("team", {}).get("id") != team_id:
                continue
            order = team_box.get("battingOrder", [])
            players = team_box.get("players", {})
            for slot_idx, pid in enumerate(order):
                p = players.get(f"ID{pid}", {})
                person = p.get("person", {})
                name = person.get("fullName")
                appearances.setdefault(pid, [name, []])
                appearances[pid][1].append(slot_idx + 1)

    ranked = sorted(appearances.items(), key=lambda kv: -len(kv[1][1]))[:9]
    lineup = []
    for pid, (name, slots) in ranked:
        avg_slot = sum(slots) / len(slots)
        lineup.append((pid, name, round(avg_slot)))
    lineup.sort(key=lambda x: x[2])
    return lineup


def get_lineup(game_pk, team_side, team_id, date_str):
    lineup = get_live_lineup(game_pk, team_side)
    if lineup:
        return lineup, "posted"
    return get_fallback_lineup(team_id, date_str), "estimated (recent starters)"


# ---------------------------------------------------------------------------
# Batter recent form
# ---------------------------------------------------------------------------

def get_recent_form(batter_id, end_date, days=15):
    """
    Hitting stats over the trailing `days` window, ending the day BEFORE
    `end_date` -- not including `end_date` itself.

    This matters most for backtesting: `end_date` is the date being tested,
    and by the time a backtest runs, that date is in the past, so MLB's API
    would otherwise include that day's own (already-known) game result in
    the "recent form" used to predict it -- a real leakage bug. Excluding
    it is also simply more correct for live use (you're never supposed to
    know today's game stats before today's game happens).
    """
    window_end = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    data = api_get(
        f"{BASE}/people/{batter_id}/stats",
        params={
            "stats": "byDateRange",
            "startDate": start_date,
            "endDate": window_end,
            "group": "hitting",
            "sportId": 1,
        },
    )
    return _extract_hitting_split(data)


# ---------------------------------------------------------------------------
# Batter vs Pitcher (BvP)
# ---------------------------------------------------------------------------

def get_bvp(batter_id, pitcher_id, seasons=None):
    """
    Career batter-vs-this-pitcher line.

    If `seasons` is given (list of ints), only sums data from those specific
    seasons instead of the live "as of now" career total. This is what makes
    backtesting leak-free: pass seasons strictly before the backtest date's
    year and the result can't contain any at-bat that happened after the
    date being tested.
    """
    if seasons:
        splits = []
        for season in seasons:
            data = api_get(
                f"{BASE}/people/{batter_id}/stats",
                params={
                    "stats": "vsPlayer",
                    "opposingPlayerId": pitcher_id,
                    "group": "hitting",
                    "sportId": 1,
                    "season": season,
                },
            )
            s = _extract_hitting_split(data)
            if s:
                splits.append(s)
        return _sum_hitting_splits(splits)

    data = api_get(
        f"{BASE}/people/{batter_id}/stats",
        params={
            "stats": "vsPlayer",
            "opposingPlayerId": pitcher_id,
            "group": "hitting",
            "sportId": 1,
        },
    )
    return _extract_hitting_split(data)


def get_vs_hand_split(batter_id, pitcher_hand, seasons=None):
    """
    Proxy for 'vs similar pitcher': batter's career line vs all pitchers
    of the same throwing hand (L or R). Used as a fallback when the direct
    BvP sample is too small to trust. `seasons` works the same as in
    get_bvp() -- pass prior seasons only for a leak-free backtest.
    """
    sit_code = "vl" if pitcher_hand == "L" else "vr"

    if seasons:
        splits = []
        for season in seasons:
            data = api_get(
                f"{BASE}/people/{batter_id}/stats",
                params={
                    "stats": "statSplits",
                    "sitCodes": sit_code,
                    "group": "hitting",
                    "sportId": 1,
                    "season": season,
                },
            )
            s = _extract_hitting_split(data)
            if s:
                splits.append(s)
        return _sum_hitting_splits(splits)

    data = api_get(
        f"{BASE}/people/{batter_id}/stats",
        params={
            "stats": "statSplits",
            "sitCodes": sit_code,
            "group": "hitting",
            "sportId": 1,
        },
    )
    return _extract_hitting_split(data)


def _sum_hitting_splits(splits):
    """Combine multiple season-level hitting splits into one aggregate dict."""
    if not splits:
        return None
    ab = sum(s["ab"] for s in splits)
    h = sum(s["h"] for s in splits)
    tb = sum(s["tb"] for s in splits)
    rbi = sum(s["rbi"] for s in splits)
    runs = sum(s["runs"] for s in splits)
    games = sum(s["games"] for s in splits)
    return {
        "ab": ab,
        "h": h,
        "avg": round(h / ab, 3) if ab else 0.0,
        "obp": None,   # not meaningfully summable without PA/BB counts; unused downstream
        "tb": tb,
        "slg": round(tb / ab, 3) if ab else 0.0,
        "rbi": rbi,
        "runs": runs,
        "games": games,
    }


def _extract_hitting_split(data):
    """Pull AB/H/AVG/OBP/SLG/TB out of a stats API response; returns dict or None."""
    if not data:
        return None
    try:
        splits = data["stats"][0]["splits"]
        if not splits:
            return None
        stat = splits[0]["stat"]
        return {
            "ab": int(stat.get("atBats", 0) or 0),
            "h": int(stat.get("hits", 0) or 0),
            "avg": float(stat.get("avg", 0) or 0),
            "obp": float(stat.get("obp", 0) or 0),
            "slg": float(stat.get("slg", 0) or 0),
            "tb": int(stat.get("totalBases", 0) or 0),
            "rbi": int(stat.get("rbi", 0) or 0),
            "runs": int(stat.get("runs", 0) or 0),
            "games": int(stat.get("gamesPlayed", 0) or 0),
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Statcast pitcher-similarity engine
# ---------------------------------------------------------------------------
#
# Approach:
#   1. Pull the target (today's opposing) pitcher's own Statcast pitch log
#      for the lookback window -> build his arsenal profile (pitch-type
#      usage %, velo, horizontal/vertical movement).
#   2. Pull the batter's own Statcast pitch log for the same window. This
#      contains every pitch he's seen, tagged with which pitcher threw it.
#      Group by opposing pitcher -> build a profile for each pitcher the
#      batter has actually faced, using the pitches thrown to him (a
#      reasonable proxy for that pitcher's arsenal, though the sample per
#      individual matchup can be small).
#   3. Score cosine similarity between each faced-pitcher's profile and the
#      target profile. Keep the ones above `similarity_threshold`,
#      excluding the target pitcher himself (he's already covered by
#      direct BvP).
#   4. Aggregate the batter's real outcomes (AB/H/BB/etc.) across all plate
#      appearances against that similar-pitcher group.
#
# Pitch buckets (collapses ~15 raw Statcast pitch_type codes into 3 groups
# so the similarity comparison isn't overly sensitive to classifier noise):
PITCH_BUCKETS = {
    "FB": {"FF", "FT", "FC", "SI"},                    # fastballs/sinkers/cutters
    "BR": {"SL", "CU", "KC", "ST", "SV", "CS"},         # breaking balls
    "OS": {"CH", "FS", "FO", "SC", "KN"},               # offspeed
}
STATCAST_HIT_EVENTS = {"single", "double", "triple", "home_run"}
TOTAL_BASE_VALUES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

# Rough league-average at-bats-per-game by lineup slot. Leadoff/2-hole hitters
# get noticeably more plate appearances over a season than the bottom of the
# order, purely from batting order mechanics. Used as a baseline to flag
# batters getting fewer at-bats than their slot would predict -- a signal of
# platooning, early pinch-hit removal, part-time play, etc. Approximate, not
# derived from this season's actual data.
EXPECTED_AB_PER_SLOT = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2, 6: 4.1, 7: 4.0, 8: 3.9, 9: 3.7}
LOW_PA_RELIABILITY_THRESHOLD = 0.75

# Shrinkage for the AB-per-game figure used IN PROJECTIONS (expected_hits,
# expected_total_bases) -- found via backtest validation (2026-07-16 to
# 2026-08-25): using the raw recent-games trailing average directly, with no
# shrinkage, caused a real and statistically robust bias. Players flagged as
# low-PA-reliability (a real, thin recent sample) BEAT their own projection by
# +0.155-0.160 on average (z=5.7-6.8, not noise) -- the model was extrapolating
# a short-term dip in playing time at full strength, when reality tends to
# regress partway back toward normal. This blends the raw recent average
# toward the slot's expected-AB baseline, trusting the raw average more as the
# recent sample (games played) grows -- full trust once recent_games reaches
# this many. NOTE: this shrinkage applies ONLY to the projection math below --
# the PA reliability FLAG itself still uses the raw (unshrunk) average, since
# that flag's whole job is to accurately surface a real recent dip, not to
# smooth it away.
AB_SHRINKAGE_FULL_TRUST_GAMES = 12

# Empirical recalibration for Runs/RBI, from backtest data (2026-07-16 to 2026-08-25).
# Both were found systematically LOW vs actual outcomes in the BvP-trusted and
# hand-split tiers specifically -- likely because the underlying BvP/hand-split
# "runs"/"rbi" rates are drawn from narrow, matchup-specific samples that don't
# reflect a full game's scoring opportunity the way recent-form's game-log rate does.
#
# IMPORTANT: a first pass at this used ONE flat global factor applied everywhere,
# fit from the overall average across all tiers. That was a mistake, caught by
# re-backtesting: the recent-form-only fallback tier (used for players with no
# BvP/hand-split history at all -- rookies, call-ups) was already close to correct
# on its own, since it doesn't touch the biased BvP/hand-split fields. Applying the
# same multiplier there anyway blew up small, noisy samples into absurd single-game
# projections (one case hit 7.12 expected runs). Factors are now tier-specific, and
# the recent-only tier gets none at all. Re-derive periodically from a fresh
# backtest rather than treating these as permanent -- and re-check the tier
# breakdown specifically, not just the overall average, before trusting a refit.
RUNS_CALIBRATION_BY_TIER = {"bvp": 2.75, "handsplit": 2.93, "recent_only": 1.0}
RBI_CALIBRATION_BY_TIER = {"bvp": 1.34, "handsplit": 1.33, "recent_only": 1.0}

# Confidence tiers for expected_combined / expected_total_bases, based on margin
# above a 1.5 line -- the most common book line for both props. Boundaries are
# empirical, from backtest hit-rate-by-margin analysis (2026-07-16 to 2026-08-25,
# n~9800 for Combined, ~3100 for Total Bases): actual clear-the-line rate rose
# monotonically with margin above 1.5 for both metrics --
#   1.5-1.8 (WEAK):      ~42% actual clear rate  -- BELOW the ~52.4% -110 breakeven
#   1.8-2.2 (MODERATE):  ~51-52%                 -- roughly at breakeven
#   2.2-2.6 (GOOD):      ~55-58%                 -- real edge
#   2.6+    (STRONG):    ~65-67%                 -- strongest edge
# "Barely clears the line" is NOT the same as "good pick" -- WEAK-tier picks
# actually underperformed a coin flip in backtesting. This is tuned to a 1.5
# line specifically; a book offering a materially different line (e.g. 2.5)
# needs its own margin bands, not an automatic rescale of these. Re-validate
# periodically against a fresh backtest rather than treating as permanent.
CONFIDENCE_TIER_LINE = 1.5
CONFIDENCE_TIER_BOUNDARIES = [
    (1.1, "STRONG"),     # value >= 2.6
    (0.7, "GOOD"),       # value >= 2.2
    (0.3, "MODERATE"),   # value >= 1.8
    (0.0, "WEAK"),       # value >= 1.5
]


def confidence_tier(value, line=CONFIDENCE_TIER_LINE):
    """Classify a projection into a confidence tier based on margin above `line`."""
    if value is None:
        return "N/A"
    margin = value - line
    for threshold, label in CONFIDENCE_TIER_BOUNDARIES:
        if margin >= threshold:
            return label
    return "BELOW LINE"

STATCAST_AB_EXCLUDE_EVENTS = {
    "walk", "hit_by_pitch", "sac_bunt", "sac_fly", "sac_fly_double_play",
    "catcher_interf", "intent_walk", "batter_interference",
}

_arsenal_cache = {}
_batter_history_cache = {}


def _bucket_for_pitch(pitch_type):
    for bucket, codes in PITCH_BUCKETS.items():
        if pitch_type in codes:
            return bucket
    return None


def build_arsenal_profile(pitch_df):
    """
    Given a Statcast pitch-level dataframe (must have pitch_type,
    release_speed, pfx_x, pfx_z columns), return a 9-dim feature vector:
    [FB_usage, FB_velo, FB_break, BR_usage, BR_velo, BR_break,
     OS_usage, OS_velo, OS_break]
    Velo/break components are scaled down by sqrt(usage) so rarely-used
    pitch types contribute little noise to the similarity comparison.
    """
    if pitch_df is None or pitch_df.empty:
        return None

    df = pitch_df.copy()
    df["bucket"] = df["pitch_type"].apply(_bucket_for_pitch)
    df = df.dropna(subset=["bucket"])
    total = len(df)
    if total == 0:
        return None

    vec = []
    for bucket in ("FB", "BR", "OS"):
        sub = df[df["bucket"] == bucket]
        usage = len(sub) / total
        if len(sub) > 0:
            velo = sub["release_speed"].mean(skipna=True) or 0.0
            break_mag = (sub["pfx_x"].abs().mean(skipna=True) or 0.0) + \
                        (sub["pfx_z"].abs().mean(skipna=True) or 0.0)
        else:
            velo, break_mag = 0.0, 0.0
        weight = math.sqrt(usage)
        vec.extend([usage, (velo / 100.0) * weight, (break_mag / 4.0) * weight])
    return vec


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_pitcher_target_profile(pitcher_id, start_date, end_date):
    """Target pitcher's own arsenal, pulled directly from his Statcast log."""
    key = (pitcher_id, start_date, end_date)
    if key in _arsenal_cache:
        return _arsenal_cache[key]
    if not STATCAST_AVAILABLE:
        return None
    try:
        df = pyb.statcast_pitcher(start_date, end_date, pitcher_id)
    except Exception as e:
        print(f"  [Statcast ERROR] pitcher {pitcher_id}: {e}")
        df = None
    profile = build_arsenal_profile(df)
    _arsenal_cache[key] = profile
    return profile


def get_batter_statcast_history(batter_id, start_date, end_date):
    """Batter's full pitch-level Statcast log for the window. Cached per batter."""
    key = (batter_id, start_date, end_date)
    if key in _batter_history_cache:
        return _batter_history_cache[key]
    if not STATCAST_AVAILABLE:
        return None
    try:
        df = pyb.statcast_batter(start_date, end_date, batter_id)
    except Exception as e:
        print(f"  [Statcast ERROR] batter {batter_id}: {e}")
        df = None
    _batter_history_cache[key] = df
    return df


def get_similar_pitcher_stats(batter_id, target_pitcher_id, target_profile,
                               start_date, end_date, similarity_threshold=0.85):
    """
    Returns dict {ab, h, avg, tb, slg, n_similar_pitchers} summarizing
    the batter's real outcomes against pitchers whose Statcast arsenal is
    similar to the target pitcher's, excluding the target pitcher himself.
    Returns None if Statcast data/pybaseball isn't available or usable.
    """
    if not STATCAST_AVAILABLE or target_profile is None:
        return None

    hist = get_batter_statcast_history(batter_id, start_date, end_date)
    if hist is None or hist.empty or "pitcher" not in hist.columns:
        return None

    similar_pitcher_ids = []
    for pid, group in hist.groupby("pitcher"):
        if pid == target_pitcher_id:
            continue
        profile = build_arsenal_profile(group)
        sim = cosine_similarity(profile, target_profile)
        if sim >= similarity_threshold:
            similar_pitcher_ids.append(pid)

    if not similar_pitcher_ids:
        return None

    # Aggregate real plate-appearance outcomes against those pitchers.
    pa_rows = hist[hist["pitcher"].isin(similar_pitcher_ids) & hist["events"].notna()]
    if pa_rows.empty:
        return None

    ab = int((~pa_rows["events"].isin(STATCAST_AB_EXCLUDE_EVENTS)).sum())
    h = int(pa_rows["events"].isin(STATCAST_HIT_EVENTS).sum())
    avg = round(h / ab, 3) if ab > 0 else 0.0
    tb = int(pa_rows["events"].map(TOTAL_BASE_VALUES).fillna(0).sum())
    slg = round(tb / ab, 3) if ab > 0 else 0.0

    return {
        "ab": ab,
        "h": h,
        "avg": avg,
        "tb": tb,
        "slg": slg,
        "n_similar_pitchers": len(similar_pitcher_ids),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rate_per_game(split, key):
    """Historical per-game rate for a counting stat (runs, rbi), or None if no games."""
    if not split or not split.get("games"):
        return None
    return split[key] / split["games"]


def _confidence_scaled_handsplit_weights(bvp_ab, min_bvp_ab, base_weights):
    """
    base_weights = (w_hand, w_recent, w_bvp), fit/chosen assuming a BvP sample
    right at the edge of min_bvp_ab (i.e. as thin as this tier gets while still
    being "present"). Applied flat, that formula treats a 1-AB BvP sample
    (nearly pure noise) the same as a 7-AB sample (still thin, but far more
    informative) -- which makes individual projections swing wildly when a
    player happens to have a tiny, extreme (e.g. 0-for-2) BvP sample.

    This scales w_bvp down linearly as bvp_ab shrinks toward 0, redistributing
    the freed weight proportionally across hand_split and recent. Continuous
    at both ends: returns base_weights unchanged at bvp_ab=min_bvp_ab, and
    drops BvP's weight to 0 (pure hand_split + recent) at bvp_ab=0.
    """
    w_hand, w_recent, w_bvp = base_weights
    confidence = max(0.0, min(1.0, bvp_ab / min_bvp_ab)) if min_bvp_ab else 1.0
    eff_bvp = w_bvp * confidence
    remaining = 1 - eff_bvp
    hand_recent_total = w_hand + w_recent
    if hand_recent_total <= 0:
        return remaining / 2, remaining / 2, eff_bvp
    return remaining * (w_hand / hand_recent_total), remaining * (w_recent / hand_recent_total), eff_bvp


def _blend_rate(bvp, hand_split, recent, key, min_bvp_ab,
                 bvp_tier_weights=(0.65, 0.35), handsplit_tier_weights=(0.45, 0.30, 0.25),
                 calibration_by_tier=None):
    """
    Same priority chain as the hit-probability blend (BvP > hand-split > recent),
    but for a raw per-game counting rate (runs or rbi) instead of a batting average.
    No Statcast-similar tier here -- the Statcast pitch log doesn't cleanly give us
    runs/RBI per PA, so that tier is skipped for this metric (falls to hand-split).

    Weights are parameterized (rather than hardcoded) because Runs and RBI were
    fit separately against backtest data and came back with different answers:
    Runs showed no meaningful improvement over the original guess (kept as-is),
    RBI's hand-split tier showed a real, holdout-validated gain -- so only RBI's
    call site below overrides handsplit_tier_weights.

    bvp_tier_weights: (w_bvp, w_recent)
    handsplit_tier_weights: (w_hand_split, w_recent, w_bvp)
    calibration_by_tier: optional dict {"bvp": x, "handsplit": y, "recent_only": z} --
    empirical multipliers applied AFTER the blend, PER TIER (not globally -- see the
    comment on RUNS_CALIBRATION_BY_TIER for why that distinction matters).
    """
    cal = calibration_by_tier or {"bvp": 1.0, "handsplit": 1.0, "recent_only": 1.0}
    bvp_rate = _rate_per_game(bvp, key)
    hand_rate = _rate_per_game(hand_split, key)
    recent_rate = _rate_per_game(recent, key) or 0.0

    if bvp and bvp["ab"] >= min_bvp_ab and bvp_rate is not None:
        w_bvp, w_recent = bvp_tier_weights
        return (w_bvp * bvp_rate + w_recent * recent_rate) * cal["bvp"]
    elif hand_split and hand_split["ab"] >= min_bvp_ab and hand_rate is not None:
        bvp_ab_n = bvp["ab"] if bvp else 0
        w_hand, w_recent, w_bvp = _confidence_scaled_handsplit_weights(bvp_ab_n, min_bvp_ab, handsplit_tier_weights)
        return (w_hand * hand_rate + w_recent * recent_rate + w_bvp * (bvp_rate or 0.0)) * cal["handsplit"]
    else:
        return recent_rate * cal["recent_only"]


def score_batter(bvp, statcast_similar, hand_split, recent, lineup_slot, min_bvp_ab=8):
    """
    Returns dict with two families of output:

    1. hit_score / run_score / rbi_score (0-100): percentile-style "how good
       does this matchup look" scores, ranked independently per category.
    2. expected_hits / expected_runs / expected_rbi / expected_combined:
       real projected COUNTS for tonight's game (e.g. 1.8), summed as
       expected_combined -- this is what actually maps to a "Hits+Runs+RBIs"
       betting prop line (usually offered as an over/under like 1.5 or 2.5),
       since that prop is a literal sum of counts, not an average of
       independent probabilities. A solo homer alone scores 3 on this
       metric (1 hit + 1 run + 1 RBI) -- that's intentional.

    Weighting (per user preference: BvP > recent form), in priority order:
        1. BvP sample >= min_bvp_ab:
               65% BvP, 35% recent
        2. Statcast-similar-pitcher sample >= min_bvp_ab (pitchers with a
           matching arsenal that this batter has actually faced):
               50% Statcast-similar, 30% recent, 20% BvP (whatever exists)
        3. Hand-split sample >= min_bvp_ab (last resort proxy):
               45% hand-split, 30% recent, 25% BvP (whatever exists)
        4. Recent form only, flagged as low-confidence.
    (Expected-count runs/rbi rates use the same priority but skip the
    Statcast tier -- see _blend_rate.)
    """
    def pct(x):
        return max(0.0, min(1.0, x)) * 100

    recent_avg = recent["avg"] if recent else 0.0
    recent_hit_rate = (recent["h"] / recent["games"]) if recent and recent.get("games") else 0.0
    bvp_avg = bvp["avg"] if bvp else 0.0

    note = ""
    if bvp and bvp["ab"] >= min_bvp_ab:
        hit_component = 0.65 * bvp["avg"] + 0.35 * recent_avg
        note = f"BvP sample trusted ({bvp['ab']} AB, {bvp['avg']:.3f})"
    elif statcast_similar and statcast_similar["ab"] >= min_bvp_ab:
        hit_component = 0.50 * statcast_similar["avg"] + 0.30 * recent_avg + 0.20 * bvp_avg
        note = (f"BvP too small ({bvp['ab'] if bvp else 0} AB) -- used Statcast-similar-arsenal "
                f"pitchers ({statcast_similar['n_similar_pitchers']} pitchers, "
                f"{statcast_similar['ab']} AB, {statcast_similar['avg']:.3f})")
    elif hand_split and hand_split["ab"] >= min_bvp_ab:
        # Weights fit from backtest data (fit_weights.py): hand_split=0.37, recent=0.18,
        # bvp=0.45 -- real, holdout-validated improvement over the original 0.45/0.30/0.25
        # guess (r 0.450 -> 0.493), BUT applied flat that treats a 1-AB BvP sample the
        # same as a 7-AB one -- confidence-scale bvp's weight by its own sample size so
        # a near-empty BvP sample doesn't swing the whole projection on noise.
        bvp_ab_n = bvp["ab"] if bvp else 0
        w_hand, w_recent, w_bvp = _confidence_scaled_handsplit_weights(bvp_ab_n, min_bvp_ab, (0.37, 0.18, 0.45))
        hit_component = w_hand * hand_split["avg"] + w_recent * recent_avg + w_bvp * bvp_avg
        note = (f"BvP/Statcast-similar too small -- used vs-hand split ({hand_split['ab']} AB, "
                f"{hand_split['avg']:.3f}) as proxy (BvP confidence: {bvp_ab_n}/{min_bvp_ab} AB)")
    else:
        hit_component = recent_avg
        note = "No usable BvP, Statcast-similar, or hand-split sample -- recent form only"

    hit_score = pct(hit_component)

    # Slugging component (for Total Bases projection) -- same tiered priority
    # as the hit-probability blend above, but tracking bases-per-AB (SLG)
    # instead of hits-per-AB (AVG), so it credits doubles/triples/HRs properly.
    recent_slg = recent["slg"] if recent else 0.0
    bvp_slg = bvp["slg"] if bvp else 0.0
    if bvp and bvp["ab"] >= min_bvp_ab:
        slg_component = 0.65 * bvp["slg"] + 0.35 * recent_slg
    elif statcast_similar and statcast_similar["ab"] >= min_bvp_ab:
        slg_component = 0.50 * statcast_similar["slg"] + 0.30 * recent_slg + 0.20 * bvp_slg
    elif hand_split and hand_split["ab"] >= min_bvp_ab:
        # Fitted weights (fit_weights.py): hand_split=0.31, recent=0.17, bvp=0.52 --
        # real improvement over the original 0.45/0.30/0.25 guess (r 0.494 -> 0.542).
        # Same confidence-scaling as hits above -- bvp=0.52 is even more aggressive,
        # so this matters more here, not less.
        bvp_ab_n2 = bvp["ab"] if bvp else 0
        w_hand2, w_recent2, w_bvp2 = _confidence_scaled_handsplit_weights(bvp_ab_n2, min_bvp_ab, (0.31, 0.17, 0.52))
        slg_component = w_hand2 * hand_split["slg"] + w_recent2 * recent_slg + w_bvp2 * bvp_slg
    else:
        slg_component = recent_slg

    # Runs/RBI 0-100 scores: blend the hit likelihood with lineup-slot tendency.
    # Slots 1-2 -> runs bias; slots 3-5 -> RBI bias; 6-9 -> both discounted.
    if lineup_slot in (1, 2):
        run_bias, rbi_bias = 1.15, 0.85
    elif lineup_slot in (3, 4, 5):
        run_bias, rbi_bias = 0.90, 1.20
    elif lineup_slot:
        run_bias, rbi_bias = 0.80, 0.80
    else:
        run_bias, rbi_bias = 1.0, 1.0

    run_score = pct((hit_component * 0.6 + recent_hit_rate * 0.4) * run_bias)
    rbi_score = pct((hit_component * 0.6 + recent_hit_rate * 0.4) * rbi_bias)

    hit_score = round(hit_score, 1)
    run_score = round(run_score, 1)
    rbi_score = round(rbi_score, 1)
    combined_score = round((hit_score + run_score + rbi_score) / 3, 1)

    # --- Real expected-count projections (the "Hits+Runs+RBIs" prop equivalent) ---
    expected_ab_for_slot = EXPECTED_AB_PER_SLOT.get(lineup_slot, 4.0)
    raw_ab_per_game = 4.0  # league-average default
    if recent and recent.get("games"):
        raw_ab_per_game = recent["ab"] / recent["games"]

    # Shrink the raw recent-games average toward the slot baseline for
    # PROJECTION purposes -- see AB_SHRINKAGE_FULL_TRUST_GAMES comment for why.
    if recent and recent.get("games"):
        shrink_confidence = min(1.0, recent["games"] / AB_SHRINKAGE_FULL_TRUST_GAMES)
    else:
        shrink_confidence = 0.0
    ab_per_game = shrink_confidence * raw_ab_per_game + (1 - shrink_confidence) * expected_ab_for_slot

    expected_hits = round(ab_per_game * hit_component, 2)
    expected_runs = round(_blend_rate(
        bvp, hand_split, recent, "runs", min_bvp_ab,
        calibration_by_tier=RUNS_CALIBRATION_BY_TIER,
    ), 2)
    # RBI hand-split tier: fit_weights.py found a real, holdout-validated improvement
    # here (r 0.416 -> 0.512) -- adopted. Runs showed no meaningful gain, left as-is.
    # Both also carry the empirical per-tier recalibration above (see constant comments
    # on RUNS_CALIBRATION_BY_TIER / RBI_CALIBRATION_BY_TIER -- NOT a flat global factor).
    expected_rbi = round(_blend_rate(
        bvp, hand_split, recent, "rbi", min_bvp_ab,
        handsplit_tier_weights=(0.11, 0.15, 0.74),  # (hand_split, recent, bvp)
        calibration_by_tier=RBI_CALIBRATION_BY_TIER,
    ), 2)
    expected_combined = round(expected_hits + expected_runs + expected_rbi, 2)
    expected_total_bases = round(ab_per_game * slg_component, 2)

    # --- Playing-time reliability flag ---
    # Not a prediction of pinch-hit substitution (that depends on bullpen
    # composition, game state, and day-of decisions we don't have access to).
    # This is a visible caution flag, not a silent adjustment to the
    # projections above: a batter getting notably fewer at-bats per game
    # than their lineup slot would predict is a real signal of platooning,
    # early removal, or part-time play -- worth knowing before trusting a
    # high projection. Uses the RAW (unshrunk) average deliberately -- this
    # flag's job is to accurately surface a real recent dip, not smooth it.
    pa_reliability = None
    low_pa_flag = False
    if recent and recent.get("games"):
        pa_reliability = round(min(1.0, raw_ab_per_game / expected_ab_for_slot), 2)
        low_pa_flag = pa_reliability < LOW_PA_RELIABILITY_THRESHOLD

    return {
        "hit_score": hit_score,
        "run_score": run_score,
        "rbi_score": rbi_score,
        "combined_score": combined_score,
        "expected_hits": expected_hits,
        "expected_runs": expected_runs,
        "expected_rbi": expected_rbi,
        "expected_combined": expected_combined,
        "expected_total_bases": expected_total_bases,
        "pa_reliability": pa_reliability,
        "low_pa_flag": low_pa_flag,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Actual results (for backtesting)
# ---------------------------------------------------------------------------

def get_game_batting_actuals(game_pk):
    """
    For a completed game, return {player_id: {h, r, rbi, ab, tb}} of what each
    batter actually did in that game. Used to score backtest predictions
    against reality.
    """
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
            batting = p.get("stats", {}).get("batting", {})
            if not pid or not batting:
                continue
            actuals[pid] = {
                "ab": int(batting.get("atBats", 0) or 0),
                "h": int(batting.get("hits", 0) or 0),
                "r": int(batting.get("runs", 0) or 0),
                "rbi": int(batting.get("rbi", 0) or 0),
                "tb": int(batting.get("totalBases", 0) or 0),
            }
    return actuals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_date(date_str, min_bvp_ab=8, recent_days=15, delay=0.15, use_statcast=False,
                  statcast_lookback_days=395, similarity_threshold=0.85, team_filter=None,
                  bvp_seasons=None, final_games_only=False, verbose=True, workers=8):
    """
    Core analysis engine, reusable by both the live CLI (run()) and
    backtest.py. Returns (rows, games) where rows is the list of per-batter
    prediction dicts and games is the list of games actually analyzed.

    `bvp_seasons`: if given (list of ints), BvP and hand-split lookups are
    restricted to those seasons only -- pass seasons strictly before the
    date being analyzed to guarantee a leak-free backtest. Leave as None
    for live/current use (uses full career-to-date).

    `final_games_only`: if True, skips any game that isn't status "Final" --
    used by the backtester so it only scores games with real results.

    `use_statcast`: OFF by default. A head-to-head backtest (2026-08-18 to
    2026-08-24, n=1,158 matched player-dates) found the Statcast-similar-
    pitcher tier performed slightly WORSE than the simpler hand-split
    fallback on both Hits (r=0.374 vs 0.394) and Total Bases (r=0.296 vs
    0.350) -- likely because the "similar pitcher" pool averaged 142
    pitchers, far too broad to represent genuine repertoire similarity. The
    code is kept (not deleted) in case a much stricter --similarity-threshold
    is worth trying later, but it's opt-in, not the default path.

    `workers`: number of batter-analysis tasks to run concurrently. This is
    I/O-bound work (waiting on the MLB Stats API / Statcast), so threading
    helps a lot -- a full slate that takes 30 min sequentially can drop to
    well under a minute. Statcast (pybaseball/Baseball Savant) calls are
    capped at a lower concurrency internally regardless of `workers`, since
    Baseball Savant is more likely to throttle/error under heavy parallel
    load than the MLB Stats API is. Lineup/pitcher-profile fetching (one
    per team per game, not per batter) stays sequential -- it's cheap and
    only needs to happen once per matchup.
    """
    def log(msg):
        if verbose:
            print(msg)

    log(f"Fetching MLB schedule for {date_str}...")
    games = get_todays_games(date_str)
    if not games:
        log("No games found (or API call failed).")
        return [], []

    if final_games_only:
        games = [g for g in games if g.get("status") == "Final"]

    if team_filter:
        needles = [n.strip().lower() for n in team_filter.split(",") if n.strip()]
        games = [
            g for g in games
            if any(n in g["home_team_name"].lower() or n in g["away_team_name"].lower() for n in needles)
        ]
        if not games:
            log(f"No games found matching --team '{team_filter}' on {date_str}.")
            return [], []

    if use_statcast and not STATCAST_AVAILABLE:
        log("pybaseball is not installed -- Statcast similarity will be skipped.")
        use_statcast = False

    statcast_start = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=statcast_lookback_days)).strftime("%Y-%m-%d")
    statcast_end = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")  # no same-day leakage

    log(f"Found {len(games)} game(s) to analyze for {date_str}.\n")

    # Statcast calls get their own, smaller concurrency cap regardless of
    # `workers` -- Baseball Savant is more fragile under heavy parallel load
    # than the MLB Stats API is.
    statcast_semaphore = threading.Semaphore(max(1, min(workers, 4)))

    # --- Phase 1: build the task list. Sequential, but cheap -- one lineup
    # fetch and one pitcher-arsenal-profile fetch per team per game, not per
    # batter. ---
    tasks = []
    for g in games:
        matchups = [
            ("away", g["away_team_id"], g["away_team_name"], g["home_pitcher"]),
            ("home", g["home_team_id"], g["home_team_name"], g["away_pitcher"]),
        ]
        for side, team_id, team_name, opp_pitcher in matchups:
            pitcher_id = opp_pitcher.get("id")
            pitcher_name = opp_pitcher.get("fullName", "TBD")
            if not pitcher_id:
                log(f"  Skipping {team_name}: no probable/starting pitcher on record.")
                continue

            pitcher_info = api_get(f"{BASE}/people/{pitcher_id}")
            pitcher_hand = "R"
            if pitcher_info:
                try:
                    pitcher_hand = pitcher_info["people"][0]["pitchHand"]["code"]
                except (KeyError, IndexError, TypeError):
                    pass

            lineup, lineup_source = get_lineup(g["gamePk"], side, team_id, date_str)
            log(f"{team_name} vs {pitcher_name} ({pitcher_hand}HP) -- lineup: {lineup_source}, {len(lineup)} batters")

            target_profile = None
            if use_statcast:
                target_profile = get_pitcher_target_profile(pitcher_id, statcast_start, statcast_end)
                if target_profile is None:
                    log(f"  [warn] No usable Statcast profile for {pitcher_name} -- falling back to hand-split only.")

            for pid, name, slot in lineup:
                if not pid:
                    continue
                tasks.append({
                    "pid": pid, "name": name, "slot": slot,
                    "gamePk": g["gamePk"],
                    "game": f"{g['away_team_name']} @ {g['home_team_name']}",
                    "team_name": team_name,
                    "pitcher_id": pitcher_id, "pitcher_name": pitcher_name, "pitcher_hand": pitcher_hand,
                    "target_profile": target_profile,
                })

    if not tasks:
        return [], games

    # --- Phase 2: run the per-batter analysis (the actual API-heavy part)
    # concurrently across all matchups for the day. ---
    def process_task(t):
        pid = t["pid"]
        bvp = get_bvp(pid, t["pitcher_id"], seasons=bvp_seasons)
        hand_split = get_vs_hand_split(pid, t["pitcher_hand"], seasons=bvp_seasons)
        recent = get_recent_form(pid, date_str, days=recent_days)

        statcast_similar = None
        if use_statcast and t["target_profile"] is not None and (not bvp or bvp["ab"] < min_bvp_ab):
            with statcast_semaphore:
                statcast_similar = get_similar_pitcher_stats(
                    pid, t["pitcher_id"], t["target_profile"], statcast_start, statcast_end,
                    similarity_threshold=similarity_threshold,
                )

        scores = score_batter(bvp, statcast_similar, hand_split, recent, t["slot"], min_bvp_ab)

        def rate(split, key):
            return round(split[key] / split["games"], 3) if split and split.get("games") else None

        return {
            "date": date_str,
            "gamePk": t["gamePk"],
            "game": t["game"],
            "batter_id": pid,
            "batter": t["name"],
            "team": t["team_name"],
            "lineup_slot": t["slot"],
            "opp_pitcher": t["pitcher_name"],
            "pitcher_hand": t["pitcher_hand"],
            # --- raw per-tier components (needed to fit blend weights against
            # actual outcomes, rather than guessing them) ---
            "bvp_ab": bvp["ab"] if bvp else 0,
            "bvp_avg": bvp["avg"] if bvp else None,
            "bvp_slg": bvp["slg"] if bvp else None,
            "bvp_runs_rate": rate(bvp, "runs"),
            "bvp_rbi_rate": rate(bvp, "rbi"),
            "statcast_similar_ab": statcast_similar["ab"] if statcast_similar else None,
            "statcast_similar_avg": statcast_similar["avg"] if statcast_similar else None,
            "statcast_similar_slg": statcast_similar["slg"] if statcast_similar else None,
            "statcast_n_similar_pitchers": statcast_similar["n_similar_pitchers"] if statcast_similar else None,
            "hand_split_ab": hand_split["ab"] if hand_split else None,
            "hand_split_avg": hand_split["avg"] if hand_split else None,
            "hand_split_slg": hand_split["slg"] if hand_split else None,
            "hand_split_runs_rate": rate(hand_split, "runs"),
            "hand_split_rbi_rate": rate(hand_split, "rbi"),
            "recent_ab": recent["ab"] if recent else None,
            "recent_games": recent["games"] if recent else None,
            "recent_avg": recent["avg"] if recent else None,
            "recent_slg": recent["slg"] if recent else None,
            "recent_runs_rate": rate(recent, "runs"),
            "recent_rbi_rate": rate(recent, "rbi"),
            # --- model outputs (unchanged) ---
            "hit_score": scores["hit_score"],
            "run_score": scores["run_score"],
            "rbi_score": scores["rbi_score"],
            "combined_score": scores["combined_score"],
            "expected_hits": scores["expected_hits"],
            "expected_runs": scores["expected_runs"],
            "expected_rbi": scores["expected_rbi"],
            "expected_combined": scores["expected_combined"],
            "expected_total_bases": scores["expected_total_bases"],
            "pa_reliability": scores["pa_reliability"],
            "low_pa_flag": scores["low_pa_flag"],
            "combined_confidence": confidence_tier(scores["expected_combined"]),
            "total_bases_confidence": confidence_tier(scores["expected_total_bases"]),
            "note": scores["note"],
        }

    rows = []
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(process_task, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                rows.append(future.result())
            except Exception as e:
                errors += 1
                log(f"  [ERROR] {t['name']} ({t['team_name']}): {e}")

    if errors:
        log(f"  {errors} batter(s) failed and were skipped -- see errors above.")

    return rows, games


def print_leaderboard(rows, score_key, title, n=10, extra_col=None):
    """Print a clean top-N table for a given score column."""
    print("\n" + "=" * 78)
    print(f"{title}  (top {n})")
    print("=" * 78)
    header = f"{'#':>2}  {'SCORE':>6}  {'BATTER':<24}{'TEAM':<18}{'OPP PITCHER':<20}"
    if extra_col:
        header += extra_col[0]
    print(header)
    print("-" * 78)
    for i, r in enumerate(sorted(rows, key=lambda x: -x[score_key])[:n], start=1):
        name = ("\u26a0 " + r["batter"]) if r.get("low_pa_flag") else r["batter"]
        line = f"{i:>2}  {r[score_key]:>6.1f}  {name:<24}{r['team']:<18}{r['opp_pitcher']:<20}"
        if extra_col:
            line += extra_col[1](r)
        print(line)


def run(date_str, min_bvp_ab, recent_days, delay, use_statcast, statcast_lookback_days,
        similarity_threshold, team_filter=None, workers=8):
    rows, games = analyze_date(
        date_str, min_bvp_ab=min_bvp_ab, recent_days=recent_days, delay=delay,
        use_statcast=use_statcast, statcast_lookback_days=statcast_lookback_days,
        similarity_threshold=similarity_threshold, team_filter=team_filter, workers=workers,
    )

    if not rows:
        print("No batters processed -- lineups may not be posted yet for this date.")
        return

    # --- Headline: expected combined count first -- this is the direct equivalent
    # of a "Hits+Runs+RBIs" betting prop (a literal sum of counts, not an average
    # of independent scores -- so a solo homer alone projects as high as 3).
    print_leaderboard(
        rows, "expected_combined", "TOP 10 PROJECTED H+R+RBI (expected combined count)",
        extra_col=("  TIER      PROJ H/R/RBI",
                   lambda r: f"  {r['combined_confidence']:<10}{r['expected_hits']:.2f}/{r['expected_runs']:.2f}/{r['expected_rbi']:.2f}"),
    )
    print("(\u26a0 = getting notably fewer at-bats per game than their lineup slot would predict --"
          " platooning/part-time play risk. Not a substitution prediction, just a caution flag.)")
    print("(TIER = confidence vs a 1.5 line, from backtested margin analysis: WEAK picks actually")
    print(" underperformed a coin flip; only GOOD/STRONG cleared the typical -110 breakeven.)")

    # --- Total Bases: its own standalone prop, projected from slugging (not part
    # of the H+R+RBI combined number above) ---
    print_leaderboard(
        rows, "expected_total_bases", "TOP 10 PROJECTED TOTAL BASES",
        extra_col=("  TIER      NOTE", lambda r: f"  {r['total_bases_confidence']:<10}{r['note']}"),
    )

    # --- Individual category leaderboards ---
    print_leaderboard(rows, "hit_score", "TOP 10 HIT PROBABILITY",
                       extra_col=("  NOTE", lambda r: f"  {r['note']}"))
    print_leaderboard(rows, "run_score", "TOP 10 RUN PROBABILITY",
                       extra_col=("  SLOT", lambda r: f"  {r['lineup_slot']}"))
    print_leaderboard(rows, "rbi_score", "TOP 10 RBI PROBABILITY",
                       extra_col=("  SLOT", lambda r: f"  {r['lineup_slot']}"))

    out_path = f"mlb_report_{date_str}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull report ({len(rows)} batters) saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB daily high-probability outcome finder")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                         help="Date to analyze, YYYY-MM-DD (default: today)")
    parser.add_argument("--min-bvp-ab", type=int, default=8,
                         help="Minimum at-bats vs a pitcher before trusting the BvP sample (default: 8)")
    parser.add_argument("--recent-days", type=int, default=15,
                         help="Trailing window for 'recent form' (default: 15 days)")
    parser.add_argument("--delay", type=float, default=0.15,
                         help="Unused now that requests run concurrently -- kept for backward "
                              "compatibility, has no effect. Use --workers to control speed/load instead.")
    parser.add_argument("--workers", type=int, default=8,
                         help="Number of batters to analyze concurrently (default: 8). Higher is "
                              "faster but hits the MLB Stats API harder. Statcast calls are capped "
                              "at min(workers, 4) internally regardless of this setting.")
    parser.add_argument("--use-statcast", action="store_true",
                         help="Opt in to Statcast pitcher-similarity analysis (OFF by default -- a head-to-head "
                              "backtest found it performs slightly worse than the simpler hand-split fallback, "
                              "and it's much slower. See analyze_date()'s docstring for the numbers.")
    parser.add_argument("--statcast-lookback-days", type=int, default=395,
                         help="How far back to pull Statcast data for arsenal/history building (default: 395, ~current + prior season)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                         help="Cosine similarity (0-1) required to count a pitcher as 'similar' (default: 0.85)")
    parser.add_argument("--team", type=str, default=None,
                         help="Scope to specific games by team name, comma-separated for multiple, "
                              "e.g. --team Dodgers,Yankees,\"Red Sox\" (matches home or away, "
                              "each match keeps that whole game -- so you'll get both teams' lineups "
                              "for any game where either side matches). Useful for a targeted run "
                              "instead of the full slate.")
    args = parser.parse_args()

    run(args.date, args.min_bvp_ab, args.recent_days, args.delay,
        use_statcast=args.use_statcast,
        statcast_lookback_days=args.statcast_lookback_days,
        similarity_threshold=args.similarity_threshold,
        team_filter=args.team,
        workers=args.workers)
